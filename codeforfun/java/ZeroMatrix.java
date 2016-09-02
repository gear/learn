import java.util.*;
import java.io.*;

public class ZeroMatrix {
  public static void main(String[] args) throws IOException {
    BufferedReader f = new BufferedReader(new FileReader(args[0]));
    StringTokenizer st = new StringTokenizer(f.readLine());
    int n = Integer.parseInt(st.nextToken());
    int m = Integer.parseInt(st.nextToken());
    int[][] matrix = new int[n][m];
    for (int i = 0; i < n; ++i) {
      st = new StringTokenizer(f.readLine());
      for (int j = 0; j < m; ++j) {
        matrix[i][j] = Integer.parseInt(st.nextToken());
      }
    }
    System.out.println("Original matrix:");
    ZeroMatrix.PrintMatrix(matrix);
    System.out.println("Boooom!");
    ZeroMatrix.ZeroDetonate(matrix);
    ZeroMatrix.PrintMatrix(matrix);
  }

  public static void ZeroDetonate(int[][] matrix) {
    int n = matrix.length;
    int m = matrix[0].length;
    int[] tmp = null;
    for (int i = 0; i < n; ++i) {
      boolean markZero = false;
      for (int j = 0; j < m; ++j) {
        if (matrix[i][j] == 0) {
          if (tmp == null) {
            tmp = matrix[i];
            for (int k = 0; k < j; ++k)
              tmp[k] = 0;
          }
          tmp[j] = 1;
          markZero = true;
        } else {
          if (markZero)
            matrix[i][j] = 0;
        }
      }
      if (markZero) {
        int j = 0;
        while (j < m && matrix[i][j] != 0)
          matrix[i][j++] = 0;
      }
    }
    if (tmp != null) {
      for (int i = 0; i < m; ++i) {
        if (tmp[i] == 1) {
          for (int j = 0; j < n; ++j)
            matrix[j][i] = 0;
        }
      }
    }
  }

  public static void PrintMatrix(int[][] matrix) {
    int n = matrix.length;
    int m = matrix[0].length;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        System.out.print(matrix[i][j] + " ");
      }
      System.out.println("");
    }
  }

}
