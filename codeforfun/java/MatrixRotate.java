import java.util.*;
import java.io.*;

public class MatrixRotate {
  public static void main(String[] args) throws IOException {
    assert args.length == 1;
    BufferedReader f = new BufferedReader(new FileReader(args[0]));
    // Read first line for matrix size
    StringTokenizer st = new StringTokenizer(f.readLine());
    int mSize = Integer.parseInt(st.nextToken());
    int[][] matrix = new int[mSize][mSize];
    for (int i = 0; i < mSize; ++i) {
      st = new StringTokenizer(f.readLine());
      for (int j = 0; j < mSize; ++j) {
        matrix[i][j] = Integer.parseInt(st.nextToken());
      }
    }
    System.out.println("Original matrix:");
    MatrixRotate.PrintMatrix(matrix);
    System.out.println("Rotated matrix:");
    MatrixRotate.InplaceRotate(matrix);
    MatrixRotate.PrintMatrix(matrix);
  }

  public static void InplaceRotate(int[][] matrix) throws NullPointerException {
    int n = matrix.length;
    if (n < 1)
      return;
    int nLayers = n / 2;
    for (int i = 0; i < nLayers; ++i) {
      for (int j = i; j < (n-1-i); ++j) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[n-1-j][i];
        matrix[n-1-j][i] = matrix[n-1-i][n-1-j];
        matrix[n-1-i][n-1-j] = matrix[j][n-1-i];
        matrix[j][n-1-i] = tmp;
      }
    }
  }

  public static void PrintMatrix(int[][] matrix) throws NullPointerException {
    for (int i = 0; i < matrix.length; ++i) {
      for (int j = 0; j < matrix.length; ++j) {
        System.out.print(matrix[i][j] + " ");
      }
      System.out.println("");
    }
  }
}
