import java.util.Arrays;

public class BadNeighbors {
  public int[] _donations;
  public int[][] _cache;

  public int maxDonation(int[] donations) {
    _donations = donations;
    _cache = new int[2][donations.length];
    Arrays.fill(_cache[0], -1);
    Arrays.fill(_cache[1], -1);
    return Math.max(max(1, 0), _donations[0] + max(2, 1));
  }

  int max(int loc, int ci) {
    if (loc >= _donations.length) return 0;
    if (loc == _donations.length-1) return ci==1 ? 0 : _donations[loc];
    if (_cache[ci][loc] != -1) return _cache[ci][loc];
    _cache[ci][loc] = Math.max(max(loc+1,ci), _donations[loc] + max(loc+2,ci));
    return _cache[ci][loc];
  }

  public static void main(String[] args) {
    BadNeighbors la = new BadNeighbors();
    int[] test = new int[] {965, 850, 698, 178, 936, 112, 944, 46, 288, 741, 23, 903, 454, 448, 539, 578, 469, 579, 32, 703, 424, 61, 488, 178, 902, 797, 933, 55, 380, 209, 791, 226, 739, 474, 431, 388, 614, 745};
    System.out.println("Testing... Expected value: " + 12773);
    System.out.println("Result: " + la.maxDonation(test));
  }
}
