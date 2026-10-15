"""
Provisions a single EC2 instance that:
  1. pulls a dataset directly from Kaggle (no local download needed),
  2. trains the byte-level BPE tokenizer on it (gpt/train_tokenizer.py),
  3. pushes the resulting tokenizer.json to Hugging Face Hub.

Secrets (Kaggle creds, HF token) are stored in SSM Parameter Store as
SecureString and fetched by the instance at boot via its IAM role -- they
are never baked into the EC2 user_data in plaintext.

Credentials (HF_TOKEN, KAGGLE_KEY, KAGGLE_USERNAME, KAGGLE_DATASET) are read
from a .env file at the repo root if present, falling back to
`pulumi config set [--secret] <key> <value>` otherwise.

Still required via `pulumi config set` (deployment settings, not secrets):
  aws:region    e.g. ap-south-1
  keyName        existing EC2 key pair name, for SSH access
  sshCidr        CIDR allowed to SSH in, e.g. "203.0.113.4/32"

Optional config (sensible defaults below):
  repoUrl, dataFile, vocabSize, sampleBytes, hfRepoId,
  instanceType, volumeSizeGb
"""

import os

import pulumi
import pulumi_aws as aws
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

config = pulumi.Config()


def secret_from_env_or_config(env_key, config_key):
    value = os.environ.get(env_key)
    return pulumi.Output.secret(value) if value else config.require_secret(config_key)


def value_from_env_or_config(env_key, config_key):
    return os.environ.get(env_key) or config.require(config_key)


key_name = config.require("keyName")
ssh_cidr = config.require("sshCidr")
kaggle_dataset = value_from_env_or_config("KAGGLE_DATASET", "kaggleDataset")

kaggle_username = secret_from_env_or_config("KAGGLE_USERNAME", "kaggleUsername")
kaggle_key = secret_from_env_or_config("KAGGLE_KEY", "kaggleKey")
hf_token = secret_from_env_or_config("HF_TOKEN", "hfToken")

repo_url = config.get("repoUrl") or "https://github.com/MorningStarTM/project-koperundevi.git"
data_file = config.get("dataFile") or "data_tamil.txt"
vocab_size = config.get_int("vocabSize") or 8000
sample_bytes = config.get_int("sampleBytes") or 200_000_000
hf_repo_id = config.get("hfRepoId") or "ErnestBeckham/gpt-v1"

# BPE training here is single-threaded, so this is a "pick fast single-core
# clock speed" choice, not a "pick many vCPUs" one -- extra cores sit idle
# until the tokenizer gets a multiprocessing rewrite.
instance_type = config.get("instanceType") or "c7i.2xlarge"
volume_size_gb = config.get_int("volumeSizeGb") or 150

ssm_prefix = "/tamil-bpe"

# --- secrets in SSM Parameter Store -----------------------------------

kaggle_username_param = aws.ssm.Parameter(
    "kaggle-username",
    name=f"{ssm_prefix}/kaggle-username",
    type="SecureString",
    value=kaggle_username,
)

kaggle_key_param = aws.ssm.Parameter(
    "kaggle-key",
    name=f"{ssm_prefix}/kaggle-key",
    type="SecureString",
    value=kaggle_key,
)

hf_token_param = aws.ssm.Parameter(
    "hf-token",
    name=f"{ssm_prefix}/hf-token",
    type="SecureString",
    value=hf_token,
)

# --- IAM role: instance can only read the three parameters above -------

region = aws.get_region().name
account_id = aws.get_caller_identity().account_id

assume_role_policy = aws.iam.get_policy_document(
    statements=[{
        "actions": ["sts:AssumeRole"],
        "principals": [{"type": "Service", "identifiers": ["ec2.amazonaws.com"]}],
    }]
)

role = aws.iam.Role(
    "bpe-trainer-role",
    assume_role_policy=assume_role_policy.json,
)

ssm_read_policy = aws.iam.RolePolicy(
    "bpe-trainer-ssm-read",
    role=role.id,
    policy=pulumi.Output.json_dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["ssm:GetParameter"],
                "Resource": [
                    f"arn:aws:ssm:{region}:{account_id}:parameter{ssm_prefix}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["kms:Decrypt"],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {"kms:ViaService": f"ssm.{region}.amazonaws.com"}
                },
            },
        ],
    }),
)

instance_profile = aws.iam.InstanceProfile(
    "bpe-trainer-profile",
    role=role.name,
)

# --- networking ---------------------------------------------------------

security_group = aws.ec2.SecurityGroup(
    "bpe-trainer-sg",
    description="Allow SSH for BPE tokenizer training instance",
    ingress=[{
        "protocol": "tcp",
        "from_port": 22,
        "to_port": 22,
        "cidr_blocks": [ssh_cidr],
    }],
    egress=[{
        "protocol": "-1",
        "from_port": 0,
        "to_port": 0,
        "cidr_blocks": ["0.0.0.0/0"],
    }],
)

ubuntu_ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical
    filters=[
        {"name": "name", "values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
        {"name": "virtualization-type", "values": ["hvm"]},
    ],
)

# --- bootstrap script -----------------------------------------------------

user_data = f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/bpe-bootstrap.log 2>&1

apt-get update -y
apt-get install -y python3-pip python3-venv git awscli

REGION="$(curl -s http://169.254.169.254/latest/meta-data/placement/region)"

KAGGLE_USERNAME=$(aws ssm get-parameter --name "{ssm_prefix}/kaggle-username" --with-decryption --query Parameter.Value --output text --region "$REGION")
KAGGLE_KEY=$(aws ssm get-parameter --name "{ssm_prefix}/kaggle-key" --with-decryption --query Parameter.Value --output text --region "$REGION")
HF_TOKEN=$(aws ssm get-parameter --name "{ssm_prefix}/hf-token" --with-decryption --query Parameter.Value --output text --region "$REGION")

mkdir -p /root/.kaggle
cat > /root/.kaggle/kaggle.json <<KAGGLE_EOF
{{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}}
KAGGLE_EOF
chmod 600 /root/.kaggle/kaggle.json

cd /root
git clone {repo_url} project
cd project

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-tokenizer.txt

mkdir -p data
kaggle datasets download -d {kaggle_dataset} -p data --unzip

cd gpt
HF_TOKEN="$HF_TOKEN" python3 train_tokenizer.py \\
  --data ../data/{data_file} \\
  --output ../models/tokenizer.json \\
  --vocab-size {vocab_size} \\
  --sample-bytes {sample_bytes} \\
  --push-to-hub \\
  --hf-repo-id {hf_repo_id}

echo "BPE training + push to hub complete" > /root/DONE
"""

instance = aws.ec2.Instance(
    "bpe-trainer",
    ami=ubuntu_ami.id,
    instance_type=instance_type,
    key_name=key_name,
    vpc_security_group_ids=[security_group.id],
    iam_instance_profile=instance_profile.name,
    associate_public_ip_address=True,
    root_block_device={
        "volume_size": volume_size_gb,
        "volume_type": "gp3",
    },
    user_data=user_data,
    tags={"Name": "tamil-bpe-trainer"},
)

pulumi.export("instance_id", instance.id)
pulumi.export("public_ip", instance.public_ip)
pulumi.export("ssh", pulumi.Output.concat("ssh -i <your-key>.pem ubuntu@", instance.public_ip))
pulumi.export("bootstrap_log", pulumi.Output.concat("ssh ... 'tail -f /var/log/bpe-bootstrap.log'"))
