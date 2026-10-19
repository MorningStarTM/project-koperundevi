"""
Tokenizer vocab-size sweep.

Provisions a single EC2 instance that trains ONE BPE tokenizer (vocab size
set per stack) on a fixed 50MB sample of data_tamil.txt (pulled from the
Kaggle "tamil-corpus" dataset), and pushes it to Hugging Face Hub as
tokenizer_<vocab_size>.json -- so multiple vocab sizes can live in the same
HF repo without overwriting each other.

Run this as 3 separate Pulumi stacks (e.g. vocab6k, vocab8k, vocab10k), each
with a different `vocabSize` config value but the SAME `sampleBytes` and
`seed` -- so all three train on the identical data slice, isolating
vocab_size as the only variable for a fair comparison.

Credentials (KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN) are read from a .env
file at the repo root if present, falling back to
`pulumi config set --secret <key> <value>` otherwise. They are stored in
SSM Parameter Store as SecureString and fetched by the instance at boot via
its IAM role -- never baked into user_data in plaintext.

Required config (set with `pulumi config set [--secret] <key> <value>`):
  aws:region     e.g. eu-north-1
  keyName         existing EC2 key pair name, for SSH access
  sshCidr         CIDR allowed to SSH in, e.g. "203.0.113.4/32"
  kaggleDataset   Kaggle dataset slug, e.g. "ernestbeckham/tamil-corpus"
  vocabSize       the vocab size THIS stack trains, e.g. 6000 / 8000 / 10000

Optional config (sensible defaults below):
  dataFile, sampleBytes, seed, hfRepoId, repoUrl, instanceType, volumeSizeGb
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


# --- config ---------------------------------------------------------------

key_name = config.require("keyName")
ssh_cidr = config.require("sshCidr")
vocab_size = config.require_int("vocabSize")

kaggle_dataset = value_from_env_or_config("KAGGLE_DATASET", "kaggleDataset")
kaggle_username = secret_from_env_or_config("KAGGLE_USERNAME", "kaggleUsername")
kaggle_key = secret_from_env_or_config("KAGGLE_KEY", "kaggleKey")
hf_token = secret_from_env_or_config("HF_TOKEN", "hfToken")

repo_url = config.get("repoUrl") or "https://github.com/MorningStarTM/project-koperundevi.git"
data_file = config.get("dataFile") or "data_tamil.txt"
hf_repo_id = config.get("hfRepoId") or "ErnestBeckham/gpt-v1"

# Validated against a real OOM: peak memory is set by --sample-bytes (not
# vocab_size), measured at ~204 bytes of Python memory per input character.
# 50MB has run clean on an 8GB instance; do not raise this casually.
sample_bytes = config.get_int("sampleBytes") or 50_000_000
seed = config.get_int("seed") or 1337

tokenizer_filename = f"tokenizer_{vocab_size}.json"

# BPE training here is single-threaded: more vCPUs don't help, faster
# per-core clock does. vocab_size doesn't change peak memory, only how many
# merge passes run, so the same instance size is safe across the whole sweep.
instance_type = config.get("instanceType") or "m7i-flex.large"
volume_size_gb = config.get_int("volumeSizeGb") or 20
ubuntu_version = config.get("ubuntuVersion") or "jammy-22.04"

stack_name = pulumi.get_stack()
ssm_prefix = f"/tamil-bpe-sweep/{stack_name}"
resource_name_tag = f"tamil-bpe-sweep-{stack_name}"

# --- secrets in SSM Parameter Store ----------------------------------------

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

# --- IAM role: instance can only read the three parameters above ----------

region = aws.get_region().name
account_id = aws.get_caller_identity().account_id

assume_role_policy = aws.iam.get_policy_document(
    statements=[{
        "actions": ["sts:AssumeRole"],
        "principals": [{"type": "Service", "identifiers": ["ec2.amazonaws.com"]}],
    }]
)

role = aws.iam.Role(
    "bpe-sweep-role",
    assume_role_policy=assume_role_policy.json,
)

# Lets us shell in via `aws ssm start-session` instead of SSH -- no inbound
# port, no IP allowlist, works regardless of the client's (unstable) public IP.
ssm_core_attachment = aws.iam.RolePolicyAttachment(
    "bpe-sweep-ssm-core",
    role=role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
)

ssm_read_policy = aws.iam.RolePolicy(
    "bpe-sweep-ssm-read",
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
    "bpe-sweep-profile",
    role=role.name,
)

# --- networking -------------------------------------------------------------

security_group = aws.ec2.SecurityGroup(
    "bpe-sweep-sg",
    description="Allow SSH for BPE tokenizer vocab-sweep training instance",
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
        {"name": "name", "values": [f"ubuntu/images/hvm-ssd/ubuntu-{ubuntu_version}-amd64-server-*"]},
        {"name": "virtualization-type", "values": ["hvm"]},
    ],
)

# --- bootstrap script ---------------------------------------------------------

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
pip install kaggle

mkdir -p data
kaggle datasets download -d {kaggle_dataset} -p data --unzip

cd gpt
HF_TOKEN="$HF_TOKEN" python3 train_tokenizer.py \\
  --data ../data/{data_file} \\
  --output ../models/{tokenizer_filename} \\
  --vocab-size {vocab_size} \\
  --sample-bytes {sample_bytes} \\
  --seed {seed} \\
  --push-to-hub \\
  --hf-repo-id {hf_repo_id}

echo "vocab={vocab_size} tokenizer training + push to hub complete" > /root/DONE
"""

instance = aws.ec2.Instance(
    "bpe-sweep",
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
    user_data_replace_on_change=True,
    tags={"Name": resource_name_tag, "Stack": stack_name, "VocabSize": str(vocab_size)},
)

pulumi.export("instance_id", instance.id)
pulumi.export("public_ip", instance.public_ip)
pulumi.export("tokenizer_filename", tokenizer_filename)
pulumi.export(
    "ssh",
    pulumi.Output.concat("ssh -i ../deploy/", key_name, ".pem ubuntu@", instance.public_ip),
)
pulumi.export(
    "bootstrap_log",
    pulumi.Output.concat(
        "ssh -i ../deploy/", key_name, ".pem ubuntu@", instance.public_ip,
        " 'tail -f /var/log/bpe-bootstrap.log'",
    ),
)
