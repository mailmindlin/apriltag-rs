package com.mindlin.apriltagrs;

import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.LongFunction;

abstract class NativeObject implements AutoCloseable {
    final ReentrantReadWriteLock ptrLock = new ReentrantReadWriteLock();
    /** Inner pointer */
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

    /**
     * Destroy native pointer
     * @param ptr Native pointer
     */
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

    static class NativeObjectReleasedException extends IllegalStateException {

    }
}
