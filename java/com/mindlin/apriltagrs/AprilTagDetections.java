package com.mindlin.apriltagrs;

import java.lang.ref.WeakReference;
import java.util.*;

import com.mindlin.apriltagrs.debug.TimeProfile;
import com.mindlin.apriltagrs.util.Matrix33;

public final class AprilTagDetections extends NativeObject implements List<AprilTagDetection>, RandomAccess {
    private static native long nativeGetFamilyPointer(long ptr, int index) throws NullPointerException;
    private static native int nativeGetTagId(long ptr, int index);
    private static native int nativeGetHammingDistance(long ptr, int index);
    private static native float nativeGetDecisionMargin(long ptr, int index);
    private static native Matrix33 nativeGetHomography(long ptr, int index);
    private static native TimeProfile nativeGetTimeProfile(long ptr);

    private final int numQuads;
    private final int count;
    private WeakReference<Optional<TimeProfile>> tp = null;
    private final Map<Long, AprilTagFamily> familyLookup;

    /**
     * Constructed by native code
     */
    private AprilTagDetections(long ptr, int numQuads, int count, Map<Long, AprilTagFamily> famlilyLookup) {
        super(ptr);
        if (count < 0)
            throw new IllegalArgumentException("Negative count");
        this.numQuads = numQuads;
        this.count = count;
        this.familyLookup = count == 0 ? null : Objects.requireNonNull(famlilyLookup);
    }

    private boolean validIndex(int index) {
        return (0 <= index && index < this.count);
    }

    private void assertValidIndex(int index) {
        if (!this.validIndex(index))
            throw new IndexOutOfBoundsException();
    }

    AprilTagFamily getDetectionFamily(int index) {
        this.assertValidIndex(index);
        long familyPtr = this.nativeRead(ptr -> nativeGetFamilyPointer(ptr, index));
        return this.familyLookup.get(familyPtr);
    }

    int getDetectionTagId(int index) {
        this.assertValidIndex(index);
        return this.nativeRead(ptr -> nativeGetTagId(ptr, index));
    }

    int getDetectionHammingDistance(int index) {
        this.assertValidIndex(index);
        return this.nativeRead(ptr -> nativeGetHammingDistance(ptr, index));
    }

    float getDetectionDecisionMargin(int index) {
        this.assertValidIndex(index);
        return this.nativeRead(ptr -> nativeGetDecisionMargin(ptr, index));
    }

    Matrix33 getDetectionHomography(int index) {
        this.assertValidIndex(index);
        return this.nativeRead(ptr -> nativeGetHomography(ptr, index));
    }

    public int getNumQuads() {
        return this.numQuads;
    }

    public Optional<TimeProfile> getTimeProfile() {
        // Try getting it from the cache (if possible)
        var tpRef = this.tp;
        if (tpRef != null) {
            var cached = tpRef.get();
            if (cached != null)
                return cached;
        }
        // It doesn't *really* matter if we build this multiple times exept for memory
        var result = this.nativeRead(ptr -> Optional.ofNullable(nativeGetTimeProfile(ptr)));
        this.tp = new WeakReference<>(result);
        return result;
    }
    
    @Override
    public int size() {
        return this.count;
    }

    @Override
    public AprilTagDetection get(int index) {
        this.assertValidIndex(index);
        //TODO: cache references?
        return new Detection(index);
    }

    @Override
    public boolean addAll(Collection<? extends AprilTagDetection> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean addAll(int index, Collection<? extends AprilTagDetection> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean isEmpty() {
        return this.count == 0;
    }

    @Override
    public boolean contains(Object o) {
        if (o instanceof Detection) {
            Detection det = (Detection) o;
            return (det.getDetections() == this) && this.validIndex(det.index);
        }
        return false;
    }

    @Override
    public Object[] toArray() {
        return this.toArray(new Detection[0]);
    }

    @Override
    public <T> T[] toArray(T[] a) {
        var result = Arrays.copyOf(a, this.size());
        for (int i = 0; i < this.size(); i++)
            result[i] = (T) new Detection(i);
        return result;
    }

    @Override
    public boolean add(AprilTagDetection aprilTagDetection) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        for (var item : c)
            if (!this.contains(item))
                return false;
        return true;
    }

    @Override
    public AprilTagDetection set(int index, AprilTagDetection element) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public void add(int index, AprilTagDetection element) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public AprilTagDetection remove(int index) {
        throw new UnsupportedOperationException("AprilTagDetections is not mutable");
    }

    @Override
    public int indexOf(Object o) {
        if (!(o instanceof AprilTagDetection))
            return -1;
        if (o instanceof Detection) {
            // O(1) lookup
            var det = (Detection) o;
            if (det.getDetections() != this)
                return -1;
            assert this.validIndex(det.index);
            return det.index;
        } else {
            //TODO
            throw new UnsupportedOperationException("AprilTagDetections.indexOf() by value");
        }
    }

    @Override
    public int lastIndexOf(Object o) {
        return this.indexOf(o);
    }

    @Override
    public Iterator<AprilTagDetection> iterator() {
        return this.listIterator();
    }

    @Override
    public ListIterator<AprilTagDetection> listIterator() {
        return this.listIterator(0);
    }

    @Override
    public ListIterator<AprilTagDetection> listIterator(int index) {
        this.assertValidIndex(index);
        return new DetectionsIterator(index);
    }

    @Override
    public List<AprilTagDetection> subList(int fromIndex, int toIndex) {
        //TODO
        return null;
    }

    @Override
    protected native void destroy(long ptr);

    private class DetectionsIterator implements ListIterator<AprilTagDetection> {
        int cursor;

        DetectionsIterator(int index) {
            this.cursor = index;
        }

        @Override
        public boolean hasNext() {
            return AprilTagDetections.this.validIndex(this.cursor);
        }

        @Override
        public AprilTagDetection next() {
            int i = this.cursor;
            try {
                AprilTagDetection result = AprilTagDetections.this.get(i);
                this.cursor = i + 1;
                return result;
            } catch (IndexOutOfBoundsException e) {
                throw new NoSuchElementException(e);
            }
        }

        @Override
        public boolean hasPrevious() {
            return this.cursor > 0;
        }

        @Override
        public AprilTagDetection previous() {
            int i = this.cursor - 1;
            try {
                return AprilTagDetections.this.get(i);
            } catch (IndexOutOfBoundsException e) {
                throw new NoSuchElementException(e);
            }
        }

        @Override
        public int nextIndex() {
            return this.cursor;
        }

        @Override
        public int previousIndex() {
            return this.cursor - 1;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("AprilTagDetections is not mutable");
        }

        @Override
        public void set(AprilTagDetection aprilTagDetection) {
            throw new UnsupportedOperationException("AprilTagDetections is not mutable");
        }

        @Override
        public void add(AprilTagDetection aprilTagDetection) {
            throw new UnsupportedOperationException("AprilTagDetections is not mutable");
        }
    }

    final class Detection extends AprilTagDetection  {
        final int index;

        Detection(int index) {
            super();
            this.index = index;
        }

        AprilTagDetections getDetections() {
            return AprilTagDetections.this;
        }
        
        @Override
        public AprilTagFamily getFamily() {
            return AprilTagDetections.this.getDetectionFamily(this.index);
        }

        @Override
        public int getTagId() {
            return AprilTagDetections.this.getDetectionTagId(this.index);
        }

        @Override
        public int getHammingDistance() {
            return AprilTagDetections.this.getDetectionHammingDistance(this.index);
        }

        @Override
        public float getDecisionMargin() {
            return AprilTagDetections.this.getDetectionDecisionMargin(this.index);
        }

        @Override
        public Matrix33 getHomography() {
            return AprilTagDetections.this.getDetectionHomography(this.index);
        }
    }
}