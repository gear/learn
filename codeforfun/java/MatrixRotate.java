import java.util.*;
import java.io.*;

public class MatrixRotate {
  public static void main(String[] args) throws IOException {
    assert args.length == 1;
    BufferedReader f = new BufferedReader(new FileReader(args[0]));
  }

  public static void InplaceRotate(int[][] matrix) throws NullPointerException {
    if (matrix.length < 1)
      return;
    int halfNCol = (matrix[0].length / 2) + (matrix[0].length % 2);
    int halfNRow = matrix.length / 2;
    for (int i = 0; i < halfNRow; ++i) {
      for (int j = 0; i < halfNCol; ++j) {
        Swap90(matrix, i, j);
      }
    }
  }

  public static void Swap90(int[][] matrix, sourceIndexi, sourceIndexj) {
    // Rotation matrix
    final int a = 0;
    final int b = 1;
    final int c = -1;
    final int d = 0;
    // Targets
    int temp = matrix[sourceIndexi][sourceIndexj];
    matrix[sourceIndexi][sourceIndexj]
  }
}
