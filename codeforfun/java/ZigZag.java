public class ZigZag {
  public static int longestZigZag(int[] sequence) {
    int n = sequence.length;
    int[] higher = new int[n];
    int[] lower = new int[n];
    int max = 0;
    for (int i = 0; i < n; ++i) {
      int max_in_lower = 0;
      int max_in_higher = 0;
      for (int j = i-1; j >= 0; --j) {
        if (sequence[i] > sequence[j] && max_in_lower < lower[j])
          max_in_lower = lower[j];
        if (sequence[i] < sequence[j] && max_in_higher < higher[j])
          max_in_higher = higher[j];
        }
      higher[i] = max_in_lower + 1;
      lower[i] = max_in_higher + 1;
      if (max < higher[i])
        max = higher[i];
      if (max < lower[i])
        max = lower[i];
    }
    return max;
  }
  public static void main(String[] args) {
    int[] arr = new int[]{1,17,5,10,13,15,10,5,16,8};
    System.out.println(ZigZag.longestZigZag(arr));
  }
}
