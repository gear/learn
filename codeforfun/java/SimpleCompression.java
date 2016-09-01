class SimpleCompression {
  public static void main(String[] args) {
    assert args.length == 1 : "Only accept one string.";
    System.out.println(SimpleCompression.BufferCompress(args[0]));
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
    if (str.length < 2):
      return;
    char cur = 0;
    char seeker = 0;
    int count = 1;
  }
}
