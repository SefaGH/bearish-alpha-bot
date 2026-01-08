import argparse
import os
import subprocess
import sys
from pathlib import Path


def ensure_host_volume_dir(path: str, *, uid: int = 1000, gid: int = 1000) -> None:
    """
    Ensure bind-mount host directories exist and are writable by the container user.

    Our Docker image runs as a non-root user (uid/gid 1000). If Docker creates the host
    mount path as root (common when the directory doesn't exist), the container can fail
    immediately with PermissionError when writing logs/data.

    Best-effort: create the directory, then chown/chmod when running as root.
    """
    if not path:
        return

    host_path = Path(path)
    try:
        host_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"??  WARNING: Failed to create host directory '{path}': {exc}", file=sys.stderr)
        return

    try:
        geteuid = getattr(os, "geteuid", None)
        if callable(geteuid) and geteuid() == 0:
            os.chown(host_path, uid, gid)
            os.chmod(host_path, 0o775)
    except Exception as exc:
        # Some filesystems (or mount options) may reject chown/chmod; don't abort startup.
        print(f"??  WARNING: Failed to set permissions on '{path}': {exc}", file=sys.stderr)


def build_docker_command(
    image: str,
    env_file: str,
    name: str = "bearish-bot",
    logs_host: str | None = None,
    data_host: str | None = None,
    detach: bool = True,
    restart_policy: str | None = "no",
    extra_env: list[str] | None = None,
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

    # All -e KEY=VAL overrides after --env-file
    if extra_env:
        for env_pair in extra_env:
            cmd.extend(["-e", env_pair])

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
        default="bearishalphabot.azurecr.io/bearish-bot:appconfig-rest-api-v2",
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
        "--force-recreate",
        default="true",
        help="If true, stop/rm existing container before run (default: true)",
    )
    parser.add_argument(
        "--just-print",
        action="store_true",
        help="Print the docker commands instead of executing them",
    )
    # New flags for env override
    parser.add_argument(
        "--debug-mode",
        type=str,
        help="Override DEBUG_MODE env (true/false/1/0/yes/no)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Override LOG_LEVEL env (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
    )
    parser.add_argument(
        "--env",
        action="append",
        help="Extra env KEY=VAL (repeatable)",
        default=[],
    )

    return parser.parse_args(argv)


def normalize_bool(val: str) -> str:
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"): return "true"
    if s in ("false", "0", "no"): return "false"
    raise ValueError(f"Invalid boolean value for --debug-mode: {val}")


def normalize_bool_flag(val: str, flag_name: str) -> bool:
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"Invalid boolean value for {flag_name}: {val}")


def container_exists(name: str) -> bool:
    result = subprocess.run(
        ["sudo", "docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return name in names


def container_running(name: str) -> bool:
    result = subprocess.run(
        [
            "sudo",
            "docker",
            "ps",
            "--filter",
            f"name=^{name}$",
            "--filter",
            "status=running",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return any(line.strip() == name for line in result.stdout.splitlines())

def validate_log_level(val: str) -> str:
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    s = str(val).strip().upper()
    if s in allowed:
        return s
    raise ValueError(f"Invalid log level for --log-level: {val}")

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    image = args.image
    env_file = args.env_file
    name = args.name

    # Validate that env_file exists
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"❌ ERROR: Environment file not found: {env_file}")
        print(f"   Please ensure the file exists before running the container.")
        return 1
    
    print(f"✓ Environment file verified: {env_file}")

    logs_host = None if args.no_volumes else args.logs_host
    data_host = None if args.no_volumes else args.data_host

    # Ensure bind-mount host dirs exist and are writable for the container user.
    if logs_host:
        ensure_host_volume_dir(logs_host)
    if data_host:
        ensure_host_volume_dir(data_host)

    # Build extra_env list
    extra_env = []
    # Generic envs first
    for env_str in args.env:
        if not env_str or "=" not in env_str:
            print(f"❌ Invalid --env format: {env_str}", file=sys.stderr)
            return 2
        k, v = env_str.split("=", 1)
        if not k or not v:
            print(f"❌ Invalid --env format: {env_str}", file=sys.stderr)
            return 2
        extra_env.append(f"{k}={v}")

    # Specific flags override generic envs if duplicate
    if args.debug_mode is not None:
        try:
            debug_val = normalize_bool(args.debug_mode)
        except Exception as e:
            print(f"❌ {e}", file=sys.stderr)
            return 2
        # Remove any previous DEBUG_MODE
        extra_env = [e for e in extra_env if not e.startswith("DEBUG_MODE=")]
        extra_env.append(f"DEBUG_MODE={debug_val}")

    if args.log_level is not None:
        try:
            log_level_val = validate_log_level(args.log_level)
        except Exception as e:
            print(f"❌ {e}", file=sys.stderr)
            return 2
        # Remove any previous LOG_LEVEL
        extra_env = [e for e in extra_env if not e.startswith("LOG_LEVEL=")]
        extra_env.append(f"LOG_LEVEL={log_level_val}")

    try:
        force_recreate = normalize_bool_flag(args.force_recreate, "--force-recreate")
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2

    exists = container_exists(name)
    running = container_running(name)

    if not force_recreate and running:
        print(f"ℹ️  Container '{name}' is already running; skipping recreate/start.")

    pull_cmd = ["sudo", "docker", "pull", image]
    stop_cmd = ["sudo", "docker", "stop", name]
    rm_cmd = ["sudo", "docker", "rm", name]
    run_cmd = build_docker_command(
        image=image,
        env_file=env_file,
        name=name,
        logs_host=logs_host,
        data_host=data_host,
        restart_policy=args.restart_policy,
        extra_env=extra_env,
    )

    commands: list[tuple[str, list[str]]] = []
    if force_recreate:
        commands.extend([
            ("Stopping existing container (if any)", stop_cmd),
            ("Removing existing container (if any)", rm_cmd),
        ])
    else:
        # If container exists but isn't running, remove it so docker run can succeed
        if exists and not running:
            commands.append(("Removing existing container (not running)", rm_cmd))

    commands.append(("Pulling image", pull_cmd))

    # Only start if forcing recreate OR container isn't already running
    if force_recreate or not running:
        commands.append(("Starting container", run_cmd))

    for description, cmd in commands:
        print(f"\n=== {description} ===")
        print(" ", " ".join(cmd))

        if args.just_print:
            continue

        # Capture output to reduce stderr noise for expected failures (stop/rm)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # stop/rm may fail if container does not exist; do not abort for those
        if result.returncode != 0:
            if description in {"Stopping existing container (if any)", "Removing existing container (if any)"}:
                print(f"  ℹ️  Container not found (expected, continuing...)")
            elif description in {"Pulling image", "Starting container"}:
                print(f"  ❌ Command failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr.strip()}")
                return result.returncode

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
