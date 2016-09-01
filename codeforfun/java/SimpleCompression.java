class SimpleCompression {
  public static void main(String[] args) {
    assert args.length == 1 : "Only accept one string.";
    System.out.println(SimpleCompression.BufferCompress(args[0]));
    System.out.println("Inline compression:");
    char[] org = args[0].toCharArray();
    System.out.println("New string length: " + SimpleCompression.InplaceCompress(org));
    System.out.println(org);
  }

  public static String BufferCompress(String str) throws NullPointerException {
    char cur = str.charAt(0);
    int count = 1;
    StringBuilder compressed = new StringBuilder();
    for (int i = 1; i < str.length(); ++i) {
      if (str.charAt(i) != cur) {
        if (count > 1) {
          compressed.append(count);
        }
        compressed.append(cur);
        cur = str.charAt(i);
        count = 1;
      } else {
        count += 1;
      }
    }
    if (count > 1)
      compressed.append(count);
    compressed.append(cur);
    return compressed.toString();
  }

  public static int InplaceCompress(char[] str) throws NullPointerException {
    if (str.length < 2)
      return str.length;
    char cur = str[0];
    int writer = 0;
    int count = 1;
    for (int i = 1; i < str.length; ++i) {
      if (str[i] != cur) {
        if (count > 1) {
          for (char c : ("" + count).toCharArray())
            str[writer++] = c;
          count = 1;
        }
        str[writer++] = cur;
        cur = str[i];
      } else {
        ++count;
      }
    }
    if (count > 1)
      for (char c : ("" + count).toCharArray())
        str[writer++] = c;
    str[writer++] = cur;
    return writer; // Number of char in new string
  }
}
