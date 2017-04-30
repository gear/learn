from __future__ import division, print_function
from sys import argv

def parse_input(input_file):
  test_cases = {}
  with open(input_file, 'r') as input:
    num_test = int(input.readline().strip())
    for t in range(num_test):
      dest, num_horse = map(int, input.readline().strip().split())
      horse_list = []
      for i in range(num_horse):
        loc, speed = map(int, input.readline().strip().split())
        horse_list.append((loc, speed))
      assert len(horse_list) == num_horse
      test_cases["Case #" + str(t+1)] = (dest, horse_list)
  return test_cases

def cruise_speed(dest, horse_list):
  arrive_time = [(dest-loc)/speed for loc,speed in horse_list]
  max_time = max(arrive_time)
  return dest/max_time    

def main():
  assert len(argv) == 3
  test_cases = parse_input(argv[1])
  out_string = ''
  for case in test_cases:
    speed = cruise_speed(*test_cases[case])
    out_string += (case + ": " + str(speed) + '\n')
  with open(argv[2], 'w') as outfile:
    outfile.write(out_string)
  print("Success!")

if __name__ == "__main__":
  main() 
