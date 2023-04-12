package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.util.Collections;
import java.util.List;

public final class Library implements Closeable {
    private static String LIBRARY_NAME = "apriltagrs";
    static {
        loadLibrary();
    }
    
    private static void loadLibrary() {
        try {
            System.loadLibrary(LIBRARY_NAME);
        } catch(UnsatisfiedLinkError e) {
            //TODO
            throw e;
        }
    }
}