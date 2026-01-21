import sys
sys.path.insert(0, 'd:/SPAG-4D')

try:
    print("Importing core...")
    from spag4d import core
    print("Core import OK")
except Exception as e:
    import traceback
    print("ERROR during import:")
    traceback.print_exc()
