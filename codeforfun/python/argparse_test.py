#!/home/hoangnt/anaconda3/bin/python

import argparse as ap

parser = ap.ArgumentParser(description='General description')
parser.add_argument('--input', nargs=1, type=str, help='input help')
parser.add_argument('--output', nargs=1, type=str, help='output help')
la = parser.parse_args()

def main():
  print(la.input)
  print(la.output)

if __name__ == '__main__':
  main()
