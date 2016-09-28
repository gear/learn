import java.util.*;

public class FlowerGarden {
    public int[] getOrdering(int[] height, int[] bloom, int[] wilt) {
        int n = height.length;
        int[] svar = new int[n];
        boolean[] used = new boolean[n];
        for (int i = 0; i < n; ++i) {
            int best = -1;
            int verdi = -1;
ytre:
            for (int j = 0; j < n; ++j) {
                if (used[j]) continue;
                if (height[j] <= verdi) continue;
                for (int k = 0; k < n; ++k) {
                    if (used[k]) continue;
                    if (height[k] >= height[j]) continue;
                    if (bloom[k] <= wilt[j] && wilt[k] >= bloom[j]) continue ytre;
                }
                best = j;
                verdi = height[j];
            }
            used[best] = true;
            svar[i] = height[best];
        }
        return svar;
    }

    public static void main(String[] args) {
        FlowerGarden la = new FlowerGarden();
        int[] height = new int[]{689, 929, 976, 79, 630, 835, 721, 386, 492, 767, 787, 387, 193, 547, 906, 642, 3, 920, 306, 735, 889, 663, 295, 892, 575, 349, 683, 699, 584, 149, 410, 710, 459, 277, 965, 112, 146, 352, 408, 432};
        int[] bloom = new int[]{4, 123, 356, 50, 57, 307, 148, 213, 269, 164, 324, 211, 249, 355, 110, 280, 211, 106, 277, 303, 63, 326, 285, 285, 269, 144, 331, 15, 296, 319, 89, 243, 254, 159, 185, 158, 81, 270, 219, 26};
        int[] wilt = new int[] {312, 158, 360, 314, 323, 343, 267, 220, 347, 197, 327, 334, 250, 360, 350, 323, 291, 323, 315, 320, 355, 334, 286, 293, 362, 181, 360, 328, 322, 344, 173, 248, 284, 301, 215, 230, 226, 331, 355, 81};
        int[] test = la.getOrdering(height, bloom, wilt);
        for (int i : test) {
            System.out.print(i + " ");
        }
        System.out.println();
    }
}
