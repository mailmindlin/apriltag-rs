package com.mindlin.apriltagrs;
import java.io.Closeable;
import java.lang.Runtime.Version;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
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
        
    }

    private AprilTagLibrary() {
        Runtime.getRuntime().version().
    }

    public native String getVersionString();

    public Version getVersion() {
        var str = this.getVersionString();
        return Version.parse(str);
    }

    
    AprilTagFamily wrapFamilyPointer(long ptr) {

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

    abstract static class NativeObject implements AutoCloseable {
        private final ReentrantReadWriteLock ptrLock = new ReentrantReadWriteLock();
        private long ptr;

        protected NativeObject(long ptr) throws NullPointerException {
            if (ptr == 0)
                throw new NullPointerException();
            this.ptr = ptr;
        }

        protected <R> R nativeRead(LongFunction<R> callback) throws IllegalStateException {
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

        protected <R> R nativeWrite(LongFunction<R> callback) {
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