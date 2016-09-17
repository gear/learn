import java.util.*;
import java.io.*;

public class DP_LCS {

    public static void main(String[] args) {
        assert (args.length == 2) : "Invalid input.";
        String str1 = args[0];
        String str2 = args[1];
        int[][] la = new int[10][10];    
        System.out.println("Hi");
    }

    public static int lcs(int i, int j, int[][] arr) {
        if (i < 0 || j < 0)
            return 0;
        else
            return arr[i][j];
    }
}
