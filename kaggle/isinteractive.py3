import os

def isinteractive():
  '''Detect user-interactive mode.'''
  return 'Interactive' == os.environ.get('KAGGLE_KERNEL_RUN_TYPE','')
