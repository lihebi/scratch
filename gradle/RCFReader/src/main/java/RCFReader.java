/**
 * Created by hebi on 6/2/17.
 */

import java.io.File;
import de.uni_bremen.st.rcf.persistence.*;
import de.uni_bremen.st.rcf.model.*;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class RCFReader {
    public static void main(String[] args) {
        System.out.println("hello");

        File file = new java.io.File("/home/hebi/.helium.d/iclones-result.rcf");



        String output = new String();

        try {
            AbstractPersistenceManager apm = PersistenceManagerFactory.getPersistenceManager(file);
            RCF rcf = apm.load(file);

            System.out.println("Loaded rcf file");

            if (rcf.getFragments().size() == 0) return;
            int sum = 0;
            // System.out.println("--- Iterating Versions ..");
            for (Version v : rcf.getVersions()) {
                // System.out.println("--- Iterating clone classes ..");
                for (CloneClass cc : v.getCloneClasses()) {
                    // System.out.println("--- Iterating Fragments ..");
                    int type = cc.getType();
                    System.out.println("-- Type " + type);
                    output += "-- Type " + type + "\n";
                    for (Fragment f : cc.getFragments()) {
                        sum = sum + f.getNumTokens();
                        SourcePosition start = f.getStart();
                        SourcePosition end = f.getEnd();
                        String filename = start.getFile().getAbsolutePath();
                        int startline = start.getLine();
                        int endline = end.getLine();
                        System.out.println("File: " + filename);
                        output += "File: " + filename + "\n";
                        System.out.println("Position: " + startline + "," + endline);
                        output += "Position: " + startline + "," + endline + "\n";
                    }
                }
            }
            // return sum / rcf.getFragments().size();

        } catch (Exception e) {
            System.out.println("Exception");
            System.out.println(e);
        }

        try {
            PrintWriter writer = new PrintWriter("/home/hebi/.helium.d/iclones-result-javaout.txt");
            writer.println(output);
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
