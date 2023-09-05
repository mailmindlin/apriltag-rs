package com.mindlin.apriltagrs;

import java.lang.ref.WeakReference;
import java.lang.reflect.Array;
import java.util.*;

import com.mindlin.apriltagrs.debug.TimeProfile;
import com.mindlin.apriltagrs.util.Matrix33;
import com.mindlin.apriltagrs.util.Point2d;

/**
 * The result of {@link AprilTagDetector#detect(byte[][])}.
 */
public final class AprilTagDetections extends NativeObject implements List<AprilTagDetection>, RandomAccess {
    private static native long nativeGetFamilyPointer(long ptr, int index) throws NullPointerException;
    private static native int nativeGetTagId(long ptr, int index);
    private static native int nativeGetHammingDistance(long ptr, int index);
    private static native double[] nativeGetCenter(long ptr, int index);
    private static native double[] nativeGetCorners(long ptr, int index);
    private static native float nativeGetDecisionMargin(long ptr, int index);
    private static native Matrix33 nativeGetHomography(long ptr, int index);
    private static native TimeProfile nativeGetTimeProfile(long ptr);

    static void subListRangeCheck(int fromIndex, int toIndex, int size) {
        if (fromIndex < 0)
            throw new IndexOutOfBoundsException("fromIndex = " + fromIndex);
        if (toIndex > size)
            throw new IndexOutOfBoundsException("toIndex = " + toIndex);
        if (fromIndex > toIndex)
            throw new IllegalArgumentException("fromIndex(" + fromIndex +
                                               ") > toIndex(" + toIndex + ")");
    }

    /**
     * Number of quads
     */
    private final int numQuads;
    /**
     * Number of tags
     */
    private final int count;
    /**
     * TimeProfile cache
     */
    private WeakReference<Optional<TimeProfile>> tp = null;
    /**
     * pointer -> AprilTagFamily lookup
     */
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

    Point2d getDetectionCenter(int index) {
        this.assertValidIndex(index);
        var buffer = this.nativeRead(ptr -> nativeGetCenter(ptr, index));
        return new Point2d(buffer[0], buffer[1]);
    }

    Point2d[] getDetectionCorners(int index) {
        this.assertValidIndex(index);
        var buffer = this.nativeRead(ptr -> nativeGetCorners(ptr, index));
        return new Point2d[] {
            new Point2d(buffer[0], buffer[1]),
            new Point2d(buffer[2], buffer[3]),
            new Point2d(buffer[4], buffer[5]),
            new Point2d(buffer[6], buffer[7]),
        };
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
        @SuppressWarnings("unchecked")
        var result = (a.length >= this.size())
            ? a
            // We prevent an arraycopy here compared with Arrays.copyOf()
            : (T[]) Array.newInstance(a.getClass().componentType(), this.size());

        for (int i = 0; i < this.size(); i++) {
            @SuppressWarnings("unchecked")
            T value = (T) new Detection(i);
            result[i] = value;
        }
        return result;
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        for (var item : c)
            if (!this.contains(item))
                return false;
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends AprilTagDetection> c) {
        throw this.immutableException();
    }

    @Override
    public boolean addAll(int index, Collection<? extends AprilTagDetection> c) {
        throw this.immutableException();
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw this.immutableException();
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw this.immutableException();
    }

    @Override
    public void clear() {
        throw this.immutableException();
    }

    @Override
    public boolean add(AprilTagDetection aprilTagDetection) {
        throw this.immutableException();
    }

    @Override
    public boolean remove(Object o) {
        throw this.immutableException();
    }

    @Override
    public AprilTagDetection set(int index, AprilTagDetection element) {
        throw this.immutableException();
    }

    @Override
    public void add(int index, AprilTagDetection element) {
        throw this.immutableException();
    }

    @Override
    public AprilTagDetection remove(int index) {
        throw this.immutableException();
    }

    private RuntimeException immutableException() {
        return new UnsupportedOperationException("AprilTagDetections is not mutable");
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
        subListRangeCheck(fromIndex, toIndex, size());
        return new SubList(fromIndex, toIndex);
    }

    @Override
    protected native void destroy(long ptr);

    /**
     * Sublist of AprilTagDetections
     */
    final class SubList extends AbstractList<AprilTagDetection> implements RandomAccess {
        private final int offset;
        protected int size;

        /**
         * Constructs a sublist of an arbitrary AbstractList, which is
         * not a SubList itself.
         */
        private SubList(int fromIndex, int toIndex) {
            this.offset = fromIndex;
            this.size = toIndex - fromIndex;
        }

        /**
         * Constructs a sublist of another SubList.
         */
        protected SubList(SubList parent, int fromIndex, int toIndex) {
            assert parent.getDetections() == AprilTagDetections.this;

            this.offset = parent.offset + fromIndex;
            this.size = toIndex - fromIndex;
        }

        private AprilTagDetections getDetections() {
            return AprilTagDetections.this;
        }

        @Override
        public AprilTagDetection get(int index) {
            Objects.checkIndex(index, size);
            return AprilTagDetections.this.get(offset + index);
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public void add(int index, AprilTagDetection element) {
            throw immutableException();
        }

        @Override
        public AprilTagDetection set(int index, AprilTagDetection element) {
            throw immutableException();
        }

        @Override
        public AprilTagDetection remove(int index) {
            throw immutableException();
        }

        @Override
        public boolean addAll(Collection<? extends AprilTagDetection> c) {
            throw immutableException();
        }

        @Override
        public boolean addAll(int index, Collection<? extends AprilTagDetection> c) {
            throw immutableException();
        }

        @Override
        public Iterator<AprilTagDetection> iterator() {
            return listIterator();
        }

        @Override
        public ListIterator<AprilTagDetection> listIterator(int index) {
            return new DetectionsIterator(this.offset, this.size, index);
        }

        @Override
        public List<AprilTagDetection> subList(int fromIndex, int toIndex) {
            subListRangeCheck(fromIndex, toIndex, size);
            return new SubList(this, fromIndex, toIndex);
        }
    }

    private class DetectionsIterator implements ListIterator<AprilTagDetection> {
        private final int offset;
        private final int size;
        int cursor;

        DetectionsIterator(int index) {
            this(0, AprilTagDetections.this.size(), index);
        }

        DetectionsIterator(int offset, int size, int index) {
            Objects.checkFromIndexSize(offset, size, index);
            this.offset = offset;
            this.size = size;
            this.cursor = index;
        }

        @Override
        public boolean hasNext() {
            return (this.cursor < this.size);
        }

        @Override
        public AprilTagDetection next() {
            if (!this.hasNext())
                throw new NoSuchElementException();
            
            int i = this.cursor;
            var result = AprilTagDetections.this.get(i + this.offset);
            this.cursor = i + 1;
            return result;
        }

        @Override
        public boolean hasPrevious() {
            return this.cursor > 0;
        }

        @Override
        public AprilTagDetection previous() {
            if (!hasPrevious())
                throw new NoSuchElementException();
            
            int i = this.cursor - 1;
            var result = AprilTagDetections.this.get(i + this.offset);
            this.cursor = i;
            return result;
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
            throw immutableException();
        }

        @Override
        public void set(AprilTagDetection aprilTagDetection) {
            throw immutableException();
        }

        @Override
        public void add(AprilTagDetection aprilTagDetection) {
            throw immutableException();
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

        @Override
        public Point2d getCenter() {
            return AprilTagDetections.this.getDetectionCenter(this.index);
        }

        @Override
        public Point2d[] getCorners() {
            return AprilTagDetections.this.getDetectionCorners(this.index);
        }
    }
}