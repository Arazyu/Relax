#!/usr/bin/env bash
#SBATCH --job-name=relax-rfc71-accept
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=/projects/b6ci/arazy/relax-rfc71-validation-20260922/validation/slurm-accept-%j.out
#SBATCH --error=/projects/b6ci/arazy/relax-rfc71-validation-20260922/validation/slurm-accept-%j.err
set -euo pipefail
ROOT="${RELAX_VALIDATION_ROOT:-/projects/b6ci/arazy/relax-rfc71-validation-20260922}"
IMAGE="${RELAX_VALIDATION_IMAGE:-/projects/b6ci/arazy/relax-inference-20260917-01a0aeec/runtime/relax-inference-sglang-0517-arm64-direct-6631103.sqsh}"
mkdir -p "$ROOT/validation"
exec /usr/bin/srun --input=none --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=1 --export=ALL \
  /usr/bin/apptainer exec --bind /projects:/projects --nv "$IMAGE" \
  bash "$ROOT/validation/relax-rfc71-acceptance-runner.sh"
