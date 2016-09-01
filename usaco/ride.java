/*
ID: Gear
LANG: JAVA
PROG: ride
TASK: ride
*/
import java.io.*;
import java.util.*;

class ride {
  public static void main(String[] args) throws IOException {
    BufferedReader f = new BufferedReader(new FileReader("ride.in"));
    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("ride.out")));
    int groupNum = 1;
    int cometNum = 1;
    int magicNumber = 47;
    for (char c : f.readLine().toCharArray()) {
      groupNum *= (c % magicNumber);
      groupNum %= magicNumber;
    }
    for (char c : f.readLine().toCharArray()) {
      cometNum *= (c % magicNumber);
      cometNum %= magicNumber;
    }
    if ((groupNum % magicNumber) == (cometNum % magicNumber))
      out.println("GO");
    else
      out.println("STAY");
    out.close();
  }

}
