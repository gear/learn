import java.util.*;
import java.io.*;

public class IncSubstring {
  public static void main(String[] args) {
    BufferedReader f = new BufferedReader(new FileReader(args[0]));

    StringTokenizer st = new StringTokenizer(f.readLine());
    int n = Integer.parseInt(st.nextToken());
    st = new StringTokenizer(f.readLine());
    int[] arr = new int[n];
    for (int i = 0; i < n; ++i)
      arr[i] = Integer.parseInt(st.nextToken());
    int lis = IncSubstring.lis(arr);
    System.out.println(lis);
  }

  public static int lis(int[] arr) {
    int[] save = new int[arr.length];
    for (int i = 0; i < arr.length; ++i) {
      for (int j = i; j >= 0; --j) {

      }
    }
  }
}
