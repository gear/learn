/*Check if string has all unique characters.
 * Created: 2016-07-08
 * Author: Hoang NT
 * v0.0: StrIsUnique.java created
 */

public class StrIsUnique {
	boolean isUnique (String str_in) {
		/*Naive approach O(n^2)
		 * Iterate through the string. 
		 * Each iteration check the current
		 * character to see duplication 
		 * until the end of the string.
		 */
		for (int i = 0; i < str_in.length()-1; ++i) {
			char curr = str_in.charAt(i);
			for (int j = i + 1; j < str_in.length(); ++j) {
				if (str_in.charAt(j) == curr)
					return false;
			}
		}
		return true;
	}

	boolean isUnique_Inline (String str_in) {
		return true;
	}

	boolean isUnique_ASCII (String str_in) {
		return true;
	}

	boolean isUnique_HashMap (String str_in) {
		return true;
	}

	public static void main(String[] args) {
		System.out.println("Cool\n");
	}
}