#!/usr/bin/env python3
"""
Tests for Live Trading Launcher GitHub Actions Workflow.

Validates the workflow configuration for PR #49 and PR #50 integration.
"""

import sys
import os
import yaml
import pytest

# Workflow file path
WORKFLOW_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    '.github',
    'workflows',
    'live_trading_launcher.yml'
)


class TestLiveTradingWorkflow:
    """Test suite for Live Trading Launcher workflow configuration."""

    @pytest.fixture
    def workflow_config(self):
        """Load workflow configuration."""
        with open(WORKFLOW_PATH, 'r') as f:
            config = yaml.safe_load(f)
            # Handle YAML 1.1 parsing of 'on:' as boolean True
            if True in config and 'on' not in config:
                config['on'] = config.pop(True)
            return config

    def test_workflow_exists(self):
        """Test workflow file exists."""
        assert os.path.exists(WORKFLOW_PATH), "Workflow file should exist"

    def test_workflow_name(self, workflow_config):
        """Test workflow has correct name."""
        assert 'name' in workflow_config
        assert 'Live Trading' in workflow_config['name']
        assert 'Ultimate Continuous Mode' in workflow_config['name']

    def test_workflow_dispatch_trigger(self, workflow_config):
        """Test workflow has manual dispatch trigger."""
        assert 'on' in workflow_config
        assert 'workflow_dispatch' in workflow_config['on']

    def test_required_inputs(self, workflow_config):
        """Test all required inputs are present."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']

        # Check required inputs
        required_inputs = [
            'mode',
            'confirm_live',
            'duration',
            'dry_run',
            'infinite',
            'auto_restart',
            'max_restarts',
            'restart_delay'
        ]

        for input_name in required_inputs:
            assert input_name in inputs, f"Input '{input_name}' should be present"

    def test_mode_input_configuration(self, workflow_config):
        """Test mode input is properly configured."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']
        mode = inputs['mode']

        assert mode['type'] == 'choice'
        assert 'paper' in mode['options']
        assert 'live' in mode['options']
        assert mode['default'] == 'paper'
        assert mode['required'] == True

    def test_infinite_mode_input(self, workflow_config):
        """Test infinite mode (Layer 1) input exists."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']

        assert 'infinite' in inputs
        assert inputs['infinite']['type'] == 'boolean'
        assert 'CONTINUOUS MODE' in inputs['infinite']['description']

    def test_auto_restart_input(self, workflow_config):
        """Test auto-restart (Layer 2) input exists."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']

        assert 'auto_restart' in inputs
        assert inputs['auto_restart']['type'] == 'boolean'
        assert 'AUTO-RESTART' in inputs['auto_restart']['description']

    def test_restart_parameters(self, workflow_config):
        """Test restart configuration parameters."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']

        # Max restarts
        assert 'max_restarts' in inputs
        assert inputs['max_restarts']['default'] == '1000'

        # Restart delay
        assert 'restart_delay' in inputs
        assert inputs['restart_delay']['default'] == '30'

    def test_required_jobs(self, workflow_config):
        """Test all required jobs are present."""
        jobs = workflow_config['jobs']

        required_jobs = [
            'validate-inputs',
            'pre-flight-checks',
            'live-trading',
            'summary'
        ]

        for job_name in required_jobs:
            assert job_name in jobs, f"Job '{job_name}' should be present"

    def test_validate_inputs_job(self, workflow_config):
        """Test validate-inputs job configuration."""
        job = workflow_config['jobs']['validate-inputs']

        assert job['runs-on'] == 'ubuntu-latest'
        assert 'outputs' in job
        assert 'should_proceed' in job['outputs']

    def test_pre_flight_checks_job(self, workflow_config):
        """Test pre-flight-checks job configuration."""
        job = workflow_config['jobs']['pre-flight-checks']

        assert job['runs-on'] == 'ubuntu-latest'
        assert 'needs' in job
        assert 'validate-inputs' in job['needs']

        # Check it has the dry-run step
        steps = job['steps']
        dry_run_steps = [s for s in steps if 'dry-run' in s.get('name', '').lower()]
        assert len(dry_run_steps) > 0, "Should have dry-run step"

    def test_live_trading_job(self, workflow_config):
        """Test live-trading job configuration."""
        job = workflow_config['jobs']['live-trading']

        assert job['runs-on'] == 'ubuntu-latest'
        assert 'needs' in job
        assert 'pre-flight-checks' in job['needs']

        # Check timeout
        assert 'timeout-minutes' in job
        assert job['timeout-minutes'] == 720  # 12 hours

    def test_environment_variables(self, workflow_config):
        """Test required environment variables are configured."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        # Find the launch step
        launch_steps = [s for s in steps if 'Launch' in s.get('name', '')]
        assert len(launch_steps) > 0, "Should have launch step"

        launch_step = launch_steps[0]
        env = launch_step.get('env', {})

        # Check BingX credentials
        assert 'BINGX_KEY' in env
        assert 'BINGX_SECRET' in env
        assert 'secrets.BINGX_KEY' in env['BINGX_KEY']
        assert 'secrets.BINGX_SECRET' in env['BINGX_SECRET']

        # Check Telegram credentials
        assert 'TELEGRAM_BOT_TOKEN' in env
        assert 'TELEGRAM_CHAT_ID' in env

    def test_launcher_script_invocation(self, workflow_config):
        """Test launcher script is properly invoked."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        # Find the launch step
        launch_steps = [s for s in steps if 'Launch' in s.get('name', '')]
        assert len(launch_steps) > 0

        launch_step = launch_steps[0]
        run_script = launch_step['run']

        # Check script path
        assert 'scripts/live_trading_launcher.py' in run_script

        # Check flags are supported
        assert '--paper' in run_script
        assert '--dry-run' in run_script
        assert '--infinite' in run_script
        assert '--auto-restart' in run_script
        assert '--max-restarts' in run_script
        assert '--restart-delay' in run_script
        assert '--duration' in run_script

    def test_artifact_uploads(self, workflow_config):
        """Test artifact uploads are configured."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        # Find artifact upload steps
        artifact_steps = [s for s in steps if s.get('uses', '').startswith('actions/upload-artifact')]

        # Should have multiple artifact uploads
        assert len(artifact_steps) >= 3, "Should have at least 3 artifact uploads"

        # Check for different artifact types
        artifact_names = [s['with']['name'] for s in artifact_steps]

        # Should upload logs, data, and health reports
        has_logs = any('log' in name.lower() for name in artifact_names)
        has_data = any('data' in name.lower() for name in artifact_names)
        has_health = any('health' in name.lower() for name in artifact_names)

        assert has_logs, "Should upload logs"
        assert has_data, "Should upload data"
        assert has_health, "Should upload health reports"

    def test_telegram_notifications(self, workflow_config):
        """Test Telegram notification steps."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        # Find Telegram notification steps
        telegram_steps = [s for s in steps if 'Telegram' in s.get('name', '')]

        # Should have success and failure notifications
        assert len(telegram_steps) >= 2, "Should have Telegram notifications"

        # Check one is for success, one for failure
        success_steps = [s for s in telegram_steps if 'Success' in s.get('name', '')]
        failure_steps = [s for s in telegram_steps if 'Failure' in s.get('name', '')]

        assert len(success_steps) > 0, "Should have success notification"
        assert len(failure_steps) > 0, "Should have failure notification"

    def test_post_session_analysis(self, workflow_config):
        """Test post-session analysis step exists."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        analysis_steps = [s for s in steps if 'Post-Session' in s.get('name', '') or 'Summary' in s.get('name', '')]
        assert len(analysis_steps) > 0, "Should have post-session analysis"

    def test_summary_job(self, workflow_config):
        """Test summary job configuration."""
        job = workflow_config['jobs']['summary']

        assert job['runs-on'] == 'ubuntu-latest'
        assert 'needs' in job
        assert 'if' in job
        assert 'always()' in job['if']

        # Check it depends on all main jobs
        assert 'validate-inputs' in job['needs']
        assert 'pre-flight-checks' in job['needs']
        assert 'live-trading' in job['needs']

    def test_safety_timeout(self, workflow_config):
        """Test 12-hour safety timeout is configured."""
        job = workflow_config['jobs']['live-trading']

        assert 'timeout-minutes' in job
        assert job['timeout-minutes'] == 720  # 12 hours = 720 minutes

    def test_python_version(self, workflow_config):
        """Test Python 3.12 is used."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        # Find setup-python step
        python_steps = [s for s in steps if 'setup-python' in s.get('uses', '')]
        assert len(python_steps) > 0

        python_step = python_steps[0]
        assert python_step['with']['python-version'] == '3.12'

    def test_configuration_display(self, workflow_config):
        """Test configuration is displayed before execution."""
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']

        # Find display configuration step
        display_steps = [s for s in steps if 'Configuration' in s.get('name', '')]
        assert len(display_steps) > 0, "Should have configuration display step"

        display_step = display_steps[0]
        run_script = display_step['run']

        # Check important configurations are displayed
        assert '100 VST' in run_script or 'Capital' in run_script
        assert 'BingX' in run_script or 'Exchange' in run_script
        assert '8' in run_script or 'Trading Pairs' in run_script


class TestWorkflowIntegration:
    """Integration tests for workflow features."""

    @pytest.fixture
    def workflow_config(self):
        """Load workflow configuration."""
        with open(WORKFLOW_PATH, 'r') as f:
            config = yaml.safe_load(f)
            # Handle YAML 1.1 parsing of 'on:' as boolean True
            if True in config and 'on' not in config:
                config['on'] = config.pop(True)
            return config

    def test_pr49_integration(self, workflow_config):
        """Test PR #49 (Live Trading Launcher) integration."""
        # Check launcher script is used
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']
        launch_step = [s for s in steps if 'Launch' in s.get('name', '')][0]

        assert 'live_trading_launcher.py' in launch_step['run']

        # Check 8 trading pairs mentioned
        config_step = [s for s in steps if 'Configuration' in s.get('name', '')][0]
        assert '8' in config_step['run']

        # Check 100 VST capital mentioned
        assert '100 VST' in config_step['run']

    def test_pr50_integration(self, workflow_config):
        """Test PR #50 (Ultimate Continuous Mode) integration."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']

        # Check Layer 1: TRUE CONTINUOUS MODE
        assert 'infinite' in inputs
        assert 'CONTINUOUS MODE' in inputs['infinite']['description']

        # Check Layer 2: AUTO-RESTART FAILSAFE
        assert 'auto_restart' in inputs
        assert 'AUTO-RESTART' in inputs['auto_restart']['description']

        # Check restart parameters
        assert inputs['max_restarts']['default'] == '1000'
        assert inputs['restart_delay']['default'] == '30'

    def test_three_layer_defense(self, workflow_config):
        """Test three-layer defense system support."""
        inputs = workflow_config['on']['workflow_dispatch']['inputs']
        job = workflow_config['jobs']['live-trading']
        steps = job['steps']
        launch_step = [s for s in steps if 'Launch' in s.get('name', '')][0]
        run_script = launch_step['run']

        # Layer 1: Infinite mode
        assert 'infinite' in inputs
        assert '--infinite' in run_script

        # Layer 2: Auto-restart
        assert 'auto_restart' in inputs
        assert '--auto-restart' in run_script

        # Layer 3: Health monitoring (automatic in launcher)
        # This is handled by the launcher script itself


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
