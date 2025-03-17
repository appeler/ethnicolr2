"""
Modern setup hooks for ethnicolr2.

This module implements hook points for setuptools using entry points,
replacing the need for custom command classes in setup.py.
"""

import os
import sys
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand
import atexit


def post_develop(command, *args, **kwargs):
    """Hook that runs after the develop command."""
    print("Running post-develop hook")
    # Add any post-development installation tasks here
    return command


def post_install(command, *args, **kwargs):
    """Hook that runs after the install command."""
    print("Running post-install hook")
    # Add any post-installation tasks here
    return command


class ToxTest(TestCommand):
    """Custom test command that runs tox."""
    user_options = [("tox-args=", "a", "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import tox
        import shlex

        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        tox.cmdline(args=args)


def register_commands(dist):
    """
    Register custom commands with setuptools.
    
    This function is called by setuptools via the entry point
    'setuptools.finalize_distribution_options'.
    """
    # Register the tox test command
    if 'test' in dist.commands:
        dist.cmdclass['test'] = ToxTest
    
    # Register hooks for develop and install commands
    if 'develop' in dist.commands:
        original_run = dist.cmdclass.get('develop', develop).run
        
        def custom_develop_run(self):
            original_run(self)
            post_develop(self)
        
        dist.cmdclass.setdefault('develop', develop).run = custom_develop_run
    
    if 'install' in dist.commands:
        original_run = dist.cmdclass.get('install', install).run
        
        def custom_install_run(self):
            original_run(self)
            post_install(self)
        
        dist.cmdclass.setdefault('install', install).run = custom_install_run


# If we need to do something at the end of installation
def finalize():
    """Run at the end of installation."""
    pass


# Register the finalize function to run at exit
atexit.register(finalize)
