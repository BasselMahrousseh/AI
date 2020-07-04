import os



#path = os.getcwd()

for root, dirs, files in os.walk("."):
        for file in files:
          if file.endswith(".mid"):
             print(os.path.join(root, file))


