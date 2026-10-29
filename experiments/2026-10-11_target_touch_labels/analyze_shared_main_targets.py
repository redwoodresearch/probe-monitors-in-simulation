"""Add a frozen shared-main-only flag outside Claude, without text output."""
from probe_shift.shared_main_targets import main

if __name__=='__main__': main()
