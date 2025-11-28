import argparse
import subprocess
import sys
from pathlib import Path


def build_docker_command(
    image: str,
    env_file: str,
    name: str = "bearish-bot",
    logs_host: str | None = None,
    data_host: str | None = None,
    detach: bool = True,
    restart_policy: str | None = "no",
) -> list[str]:
    cmd: list[str] = [
        "sudo",
        "docker",
        "run",
    ]

    if detach:
        cmd.append("-d")

    cmd.extend([
        "--name",
        name,
        "--env-file",
        env_file,
    ])

    if restart_policy and restart_policy != "no":
        cmd.extend(["--restart", restart_policy])

    if logs_host:
        cmd.extend(["-v", f"{logs_host}:/app/logs"])
    if data_host:
        cmd.extend(["-v", f"{data_host}:/app/data"])

    cmd.append(image)
    return cmd


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Utility to stop, remove and (re)start the bearish-bot container on the VM "
            "with a consistent docker run command. Intended for manual ops only."
        )
    )

    parser.add_argument(
        "--image",
        default="bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-4",
        help="Docker image to run (default: %(default)s)",
    )
    parser.add_argument(
        "--env-file",
        default="/home/azureuser/bearish-bot.env",
        help="Path to env file on VM (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        default="bearish-bot",
        help="Container name (default: %(default)s)",
    )
    parser.add_argument(
        "--logs-host",
        default="/mnt/bearish/logs",
        help="Host path for logs volume (default: %(default)s)",
    )
    parser.add_argument(
        "--data-host",
        default="/mnt/bearish/data",
        help="Host path for data volume (default: %(default)s)",
    )
    parser.add_argument(
        "--no-volumes",
        action="store_true",
        help="Do not mount host volumes (ignores logs-host and data-host)",
    )
    parser.add_argument(
        "--restart-policy",
        default="no",
        choices=("no", "always", "unless-stopped", "on-failure"),
        help="Docker restart policy; defaults to no auto-restart",
    )
    parser.add_argument(
        "--just-print",
        action="store_true",
        help="Print the docker commands instead of executing them",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    image = args.image
    env_file = args.env_file
    name = args.name

    logs_host = None if args.no_volumes else args.logs_host
    data_host = None if args.no_volumes else args.data_host

    stop_cmd = ["sudo", "docker", "stop", name]
    rm_cmd = ["sudo", "docker", "rm", name]
    pull_cmd = ["sudo", "docker", "pull", image]
    run_cmd = build_docker_command(
        image=image,
        env_file=env_file,
        name=name,
        logs_host=logs_host,
        data_host=data_host,
        restart_policy=args.restart_policy,
    )

    commands = [
        ("Stopping existing container (if any)", stop_cmd),
        ("Removing existing container (if any)", rm_cmd),
        ("Pulling image", pull_cmd),
        ("Starting container", run_cmd),
    ]

    for description, cmd in commands:
        print(f"\n=== {description} ===")
        print(" ", " ".join(cmd))

        if args.just_print:
            continue

        result = subprocess.run(cmd)
        # stop/rm may fail if container does not exist; do not abort for those
        if description in {"Pulling image", "Starting container"} and result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}")
            return result.returncode

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
