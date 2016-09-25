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
  public static int longestZigZag(int[] sequence, boolean fast) {
    int n = sequence.length;
    if (n == 1) return 1;
    int[] distances = new int[n-1];
    int fstNonzero = 0;
    for (int i = n-1; i >= 1; --i) {
      distances[i-1] = sequence[i] - sequence[i-1];
      if (distances[i-1] != 0)
        fstNonzero = i-1;
    }
    int root = distances[fstNonzero];
    int len = 2;
    for (int i = fstNonzero + 1; i < n-1; ++i) {
      if (distances[i] * root < 0) {
        root = -root;
        ++len;
      }
    }
    return len;
  }
  public static void main(String[] args) {
    int[] arr = new int[]{1,17,5,10,13,15,10,5,16,8};
    System.out.println(ZigZag.longestZigZag(arr));
    System.out.println(ZigZag.longestZigZag(arr, true));
  }
}
