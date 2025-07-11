
import os
for root, _, files in os.walk('.'):
    for f in files:
        print(os.path.relpath(os.path.join(root,f), '.'))

