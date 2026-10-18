"""
Phase 2 -- provision with Pulumi.

Provisions a single CPU EC2 instance that:
  STEP 0. installs Prometheus + Grafana, scraping gpt/metrics.py's endpoint,
  STEP 1. pulls data/literature.txt from the HF dataset repo (no Kaggle),
  STEP 2. downloads the already-trained tokenizer.json from ErnestBeckham/gpt-v1
          (the tokenizer itself was trained and pushed locally -- not here).

Training (gpt/run_train.sh) is NOT auto-started by this script -- SSH in and
run it by hand, same as you did locally, once the instance is up.

The HF token is stored in SSM Parameter Store as a SecureString and fetched
by the instance at boot via its IAM role -- it is never baked into the EC2
user_data in plaintext. Read from a .env file at the repo root if present,
falling back to `pulumi config set --secret hfToken <value>` otherwise.

Still required via `pulumi config set` (deployment settings, not secrets):
  aws:region    e.g. ap-south-1
  keyName        existing EC2 key pair name, for SSH access
  sshCidr        CIDR allowed to SSH in, e.g. "203.0.113.4/32"

Optional config (sensible defaults below):
  repoUrl, hfDatasetRepo, dataFile, hfRepoId, tokenizerFilename,
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


# --- config: deployment settings + secrets ------------------------------

key_name = config.require("keyName")
ssh_cidr = config.require("sshCidr")
hf_token = secret_from_env_or_config("HF_TOKEN", "hfToken")

repo_url = config.get("repoUrl") or "https://github.com/MorningStarTM/project-koperundevi.git"
hf_dataset_repo = config.get("hfDatasetRepo") or "ErnestBeckham/tamil-literature"
data_file = config.get("dataFile") or "literature.txt"
hf_repo_id = config.get("hfRepoId") or "ErnestBeckham/gpt-v1"
tokenizer_filename = config.get("tokenizerFilename") or "tokenizer_literature.json"

instance_type = config.get("instanceType") or "c7i.2xlarge"
volume_size_gb = config.get_int("volumeSizeGb") or 150
ubuntu_version = config.get("ubuntuVersion") or "jammy-22.04"

stack_name = pulumi.get_stack()
ssm_prefix = f"/tamil-bpe/{stack_name}"
resource_name_tag = f"tamil-bpe-trainer-{stack_name}"

# --- block 1: secret in SSM Parameter Store -----------------------------

hf_token_param = aws.ssm.Parameter(
    "hf-token",
    name=f"{ssm_prefix}/hf-token",
    type="SecureString",
    value=hf_token,
)

# --- block 2: IAM role, scoped to read only the parameter above --------

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

# Lets us shell in via `aws ssm start-session` instead of SSH -- no inbound
# port, no IP allowlist, works regardless of the client's (unstable) public IP.
ssm_core_attachment = aws.iam.RolePolicyAttachment(
    "bpe-trainer-ssm-core",
    role=role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
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

# --- block 3: networking -------------------------------------------------

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
        {"name": "name", "values": [f"ubuntu/images/hvm-ssd/ubuntu-{ubuntu_version}-amd64-server-*"]},
        {"name": "virtualization-type", "values": ["hvm"]},
    ],
)

# --- block 4: bootstrap script, run once at boot -------------------------
# STEP 0 (dashboard stack) -> STEP 1 (data) -> STEP 2 (tokenizer download)
# Training itself (run_train.sh) is NOT auto-started here -- you SSH in and
# kick it off by hand, same as locally, so nothing runs (and nothing is
# billed for GPU-less CPU time) without you choosing to start it.

user_data = f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/bpe-bootstrap.log 2>&1

apt-get update -y
apt-get install -y python3-pip python3-venv git awscli prometheus

# STEP 0: Grafana (not in Ubuntu's default repos) + a scrape config pointing
# at the trainer's /metrics endpoint, same setup as the local one.
apt-get install -y apt-transport-https software-properties-common wget
wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" \\
  > /etc/apt/sources.list.d/grafana.list
apt-get update -y
apt-get install -y grafana

cat > /etc/prometheus/prometheus.yml <<'PROM_EOF'
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
  - job_name: tamil-gpt-training
    static_configs:
      - targets: ["localhost:8000"]
PROM_EOF

systemctl enable --now prometheus
systemctl enable --now grafana-server

REGION="$(curl -s http://169.254.169.254/latest/meta-data/placement/region)"
HF_TOKEN=$(aws ssm get-parameter --name "{ssm_prefix}/hf-token" --with-decryption --query Parameter.Value --output text --region "$REGION")

cd /root
git clone {repo_url} project
cd project

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-tokenizer.txt
pip install -r requirements.txt

# STEP 1: pull the dataset from the HF dataset repo (not Kaggle)
mkdir -p data
HF_TOKEN="$HF_TOKEN" python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='{hf_dataset_repo}', repo_type='dataset', filename='{data_file}')
shutil.copy(path, 'data/{data_file}')
"

# STEP 2: the tokenizer is already trained and pushed (ErnestBeckham/gpt-v1)
# -- download it rather than retraining it from scratch.
mkdir -p models
HF_TOKEN="$HF_TOKEN" python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='{hf_repo_id}', filename='{tokenizer_filename}')
shutil.copy(path, 'models/{tokenizer_filename}')
"

echo "dashboard stack + data pull + tokenizer download complete -- SSH in and run gpt/run_train.sh when ready" > /root/DONE
"""

# --- block 5: the instance itself, plus outputs ---------------------------

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
    user_data_replace_on_change=True,
    tags={"Name": resource_name_tag, "Stack": stack_name},
)

pulumi.export("instance_id", instance.id)
pulumi.export("public_ip", instance.public_ip)
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
