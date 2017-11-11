#!/home/hoangnt/anaconda3/bin/python

import argparse as ap

parser = ap.ArgumentParser(description='General description')
parser.add_argument('--input', type=str, help='input help')
parser.add_argument('--output', type=str, help='output help', default=None)
la = parser.parse_args()

def main():
  print(la.input)
  print(la.output)

if __name__ == '__main__':
  main()
