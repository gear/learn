import java.util.*;
import java.io.*;

public class IncSubstring {
  public static void main(String[] args) throws IOException {
    BufferedReader f = new BufferedReader(new FileReader(args[0]));

    StringTokenizer st = new StringTokenizer(f.readLine());
    int n = Integer.parseInt(st.nextToken());
    st = new StringTokenizer(f.readLine());
    int[] arr = new int[n];
    for (int i = 0; i < n; ++i) 
      arr[i] = Integer.parseInt(st.nextToken());
    int lis = IncSubstring.lis(arr);
    System.out.println("Longest length: " + lis);
  }

  public static int lis(int[] arr) {
    int[] save = new int[arr.length];
    int globalmax = 1;
    for (int i = 0; i < arr.length; ++i) {
      int max = 0;
      for (int j = i; j >= 0; --j) {
        if (arr[j] <= arr[i] && save[j] > max)
          max = save[j];
      }
      save[i] = max + 1;
      if (save[i] > globalmax) {
        globalmax = save[i];
      }
    }
    return globalmax;
  }
}
