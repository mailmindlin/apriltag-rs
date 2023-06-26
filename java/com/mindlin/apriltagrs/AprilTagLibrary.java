package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.lang.Runtime.Version;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.PosixFileAttributes;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.LongFunction;

/**
 * Native library manager
 */
public final class AprilTagLibrary {
    private static String LIBRARY_NAME = "apriltagrs";
    
    static void loadLibrary() {
        try {
            System.loadLibrary(LIBRARY_NAME);
        } catch(UnsatisfiedLinkError e) {
            //TODO
            throw e;
        }
    }

    private static AprilTagLibrary instance = null;

    public static AprilTagLibrary getInstance() {
        if (instance != null)
            return instance;
        synchronized (AprilTagLibrary.class) {
            if (instance == null)
                instance = new AprilTagLibrary();
        }
        return instance;
    }

    private static Path createTempFile(String name) throws IOException {
        Objects.requireNonNull(name);
        // Generate temporary file
        var tempDirStr = System.getProperty("java.io.tmpdir");
        var tempDir = tempDirStr == null ? null : Paths.get(tempDirStr);

        var prefix = name;
        var suffix = "";
        var dotIdx = name.lastIndexOf('.');
        if (dotIdx > 0) {
            prefix = name.substring(0, dotIdx);
            suffix = name.substring(dotIdx);
        }
        
        while (true) {
            try {
                if (tempDir == null)
                    return Files.createTempFile(prefix, suffix);
                else
                    return Files.createTempFile(tempDir, prefix, suffix);
            } catch (IllegalArgumentException e) {
                if (prefix != null || suffix != null) {
                    // Try again with no prefix/suffix
                    prefix = null;
                    suffix = null;
                } else {
                    throw e;
                }
            } catch (IOException e) {
                if (tempDir != null && !Files.isDirectory(tempDir)) {
                    tempDir = null;
                } else {
                    throw e;
                }
            }
        }
    }

    private static Path extractFromJar(String name) {
        Path tempFile = null;
        try (var is = AprilTagLibrary.class.getResourceAsStream("/lib/" + name)) {
            // Check if it's really there
            if (is == null)
                return null;
            
            try {
                tempFile = createTempFile(name);
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
            
            try (var os = Files.newOutputStream(tempFile, StandardOpenOption.CREATE_NEW)) {
                byte[] buffer = new byte[4096];
                for (int len; (len = is.read(buffer, 0, buffer.length)) >= 0;) {
                    os.write(buffer, 0, len);
                }
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tempFile;
    }

    private static Path loadFromJar(String name) {
        var tempPath = extractFromJar(name);
        if (tempPath == null)
            return null;
        var absPath = tempPath.toAbsolutePath();
        try {
            System.load(tempPath.toString());
        } catch (UnsatisfiedLinkError e) {
            try {
                Files.delete(absPath);
            } catch (IOException e1) {}
            return null;
        }
        return absPath;
    }

    private static Path loadFromJar(String arch, String vendor, String os, String ext) {

    }

    Path path;
    private AprilTagLibrary() {
        try {
            System.loadLibrary(LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
        }

        var osName = System.getProperty("os.name");
        var osArch = System.getProperty("os.arch");
        if (osName.contains("win")) {
            final String ext = "dll";
            if ((this.path = loadFromJar(LIBRARY_NAME + "." + ext)) != null)
                return;
            if ((this.path = loadFromJar(osArch, "pc", "windows", ext)) != null)
                return;
        } else if (osName.contains("mac")) {
            final String ext = "dylib";
            if ((this.path = loadFromJar(LIBRARY_NAME + "." + ext)) != null)
                return;
            if ((this.path = loadFromJar(osArch, "apple", "darwin", ext)) != null)
                return;
        } else {

        }
    }

    public native String getVersionString();

    public Version getVersion() {
        var str = this.getVersionString();
        return Version.parse(str);
    }

    /**
     * Massage ByteBuffer so it's more easily accessed by native code
     */
    static ByteBuffer massageBuffer(ByteBuffer buf) {
        Objects.requireNonNull(buf);

        if (buf.isDirect())
            return buf;

        if (buf.hasArray()) {
            try {
                buf.array();
                buf.arrayOffset();
                return buf;
            } catch (ReadOnlyBufferException | UnsupportedOperationException e) {
            }
        }
    }

    abstract static class Cache<T> {
        private final ConcurrentHashMap<Long, WeakReference<T>> lookup = new ConcurrentHashMap<>();
        
        abstract T wrap(long ptr);
    }

    static class NativeObjectReleasedException extends IllegalStateException {

    }

    abstract static class NativeObject implements AutoCloseable {
        final ReentrantReadWriteLock ptrLock = new ReentrantReadWriteLock();
        long ptr;

        protected NativeObject(long ptr) throws NullPointerException {
            if (ptr == 0)
                throw new NullPointerException();
            this.ptr = ptr;
        }

        protected <R> R nativeRead(LongFunction<R> callback) throws NativeObjectReleasedException {
            var readLock = NativeObject.this.ptrLock.readLock();
            readLock.lock();
            try {
                long ptr = this.ptr;
                if (ptr == 0)
                    throw new IllegalStateException("Use after close");
                return callback.apply(ptr);
            } finally {
                readLock.unlock();
            }
        }

        protected <R> R nativeWrite(LongFunction<R> callback) throws NativeObjectReleasedException {
            var writeLock = NativeObject.this.ptrLock.writeLock();
            writeLock.lock();
            try {
                long ptr = this.ptr;
                if (ptr == 0)
                    throw new IllegalStateException("Use after close");
                return callback.apply(ptr);
            } finally {
                writeLock.unlock();
            }
        }

        protected abstract void destroy(long ptr);

        @Override
        public void close() {
            var writeLock = ptrLock.writeLock();
            writeLock.lock();
            try {
                long ptr = this.ptr;
                this.ptr = 0;
                this.destroy(ptr);
            } finally {
                writeLock.unlock();
            }
        }
    }
}