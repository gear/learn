import java.util.*;
import java.io.*;

public class CoinSum {

  int[] coinVals; 
  int[] saves;

  CoinSum(int[] coinVals) {
    this.coinVals = coinVals;
  }

  public static void main(String[] args) throws IOException {
    assert args.length == 1 : "Please give a contains 2 lines: n and v_i.";
    BufferedReader f = new BufferedReader(new FileReader(args[0]));
    // Read the first line with format <numVals> <sum>
    StringTokenizer st = new StringTokenizer(f.readLine());
    int numVals = Integer.parseInt(st.nextToken());
    int s = Integer.parseInt(st.nextToken());
    // Read <numVals> values in the next line
    st = new StringTokenizer(f.readLine());
    int[] coinVals = new int[numVals];
    for (int i = 0; i < numVals; ++i) {
      coinVals[i] = Integer.parseInt(st.nextToken());
    }
    CoinSum prob = new CoinSum(coinVals);
    System.out.println("Minimum number of coins to get " + s + " is: " + prob.minCoins(s)); 
  }

  public int minCoins(int s) {
    System.out.println("Call: " + s);
    if (this.saves == null)
      this.saves = new int[s+1];
    if (s < 0)
      return -1;
    if (s == 0)
      return 0;
    if (this.saves[s] > 0)
      return this.saves[s];
    int min = Integer.MAX_VALUE;
    for (int val : this.coinVals) {
      int cnum = this.minCoins(s - val);
      if (cnum < 0)
        continue;
      if (cnum < min)
        min = cnum;
    }
    this.saves[s] = 1 + min;
    return 1 + min;
  }
}
