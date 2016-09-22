import java.util.*;
import java.io.*;

public class RGB {
  public static void main(String[] args) {
    char[] la = new char[]{'B', 'R', 'G', 'R', 'B', 'G', 'G', 'B', 'R'};
    System.out.println(RGB.minElement(la));
  }
  public static int minElement(char[] sequence) {
    int rCount = 0;
    int gCount = 0;
    int bCount = 0;
    for (char c : sequence) {
      switch (c) {
        case 'R': rCount++; break;
        case 'G': gCount++; break;
        case 'B': bCount++; break;
        default: break;
      }
    }
    int sum = rCount + gCount + bCount;
    if (sum == rCount || sum == gCount || sum == bCount)
      return sequence.length;
    rCount = rCount % 2;
    bCount = bCount % 2;
    gCount = gCount % 2;
    if (rCount * bCount * gCount == 1 || rCount + gCount + bCount == 0)
      return 2;
    return 1;
  }
}
