package com.mindlin.apriltagrs;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;

import com.mindlin.apriltagrs.util.Image2D;

/**
 * A detector for AprilTags
 */
public final class AprilTagDetector extends NativeObject {
    /**
     * Create new detector object
     * 
     * @return Handle to detector
     */
    private static native long nativeCreate(
        // Config values
        int nthreads, float quadDecimate, float quadSigma, boolean refineEdges, float decodeSharpening, String debugPath,
        // QTP values
        int minClusterPixels, int maxNumMaxima, float cosCriticalRad, float maxLineFitMSE, int minWhiteBlackDiff, boolean deglitch,
        long[] familyPtrs, int[] familyHammings
    );
    private static native AprilTagDetections nativeDetect(long ptr, ByteBuffer buf, int width, int height, int stride, Map<Long, AprilTagFamily> familyLookup);

    public static class Builder implements Cloneable {
        private Config config;
        private QuadThresholdParameters qtp;
        private Map<AprilTagFamily, Integer> families;

        public Builder() {
            this.config = new Config();
            this.qtp = new QuadThresholdParameters();
            this.families = Collections.emptyMap();
        }

        public Builder(AprilTagDetector det) {
            Objects.requireNonNull(det);
            this.config = new Config(det.config);
            this.qtp = new QuadThresholdParameters(det.qtp);
            this.families = new LinkedHashMap<>(det.families);
        }

        /**
         * Copy constructor
         */
        public Builder(Builder src) {
            this(Objects.requireNonNull(src).config, src.qtp, src.families, true);
        }

        private Builder(Config config, QuadThresholdParameters qtp, Map<AprilTagFamily, Integer> families, boolean clone) {
            Objects.requireNonNull(config, "config");
            Objects.requireNonNull(qtp, "QuadThresholdParameters");
            Objects.requireNonNull(families, "families");

            if (clone) {
                // Validate and clone
                for (var entry : families.entrySet()) {
                    var family = entry.getKey();
                    Objects.requireNonNull(family, "family");
                    var bitsCorrected = entry.getValue();
                    // Null value == default
                    if (bitsCorrected == null)
                        continue;
                    
                    family.validateBitsCorrected(bitsCorrected.intValue());
                }
                this.config = new Config(config);
                this.qtp = new QuadThresholdParameters(qtp);
                this.families = new LinkedHashMap<>(families);
            } else {
                this.config = config;
                this.qtp = qtp;
                this.families = families;
            }
        }

        public Config getConfig() {
            return this.config;
        }

        public Builder setConfig(Config config) {
            Objects.requireNonNull(config);
            if (config != this.config)
                this.config = new Config(config);
            return this;
        }

        public QuadThresholdParameters getQuadThresholdParameters() {
            return this.qtp;
        }

        public Builder setQuadThresholdParameters(QuadThresholdParameters qtp) {
            Objects.requireNonNull(qtp);
            if (qtp != this.qtp)
                this.qtp = new QuadThresholdParameters(qtp);
            return this;
        }

        @Override
        public Builder clone() {
            return new Builder(this.config, this.qtp, this.families, true);
        }

        private Builder mapConfig(Consumer<Config> update) {
            // var config = new Config(this.config);
            update.accept(this.config);
            // return new Builder(config, this.qtp, this.families, true);
            return this;
        }

        /**
         * Set number of threads to use
         * @param nthreads Number of threads. 0 means 
         * @return modified builder
         */
        public Builder numThreads(int nthreads) {
            return this.mapConfig(config -> config.setNumThreads(nthreads));
        }

        /**
         * Adds an AprilTag family to detect, with a default hamming distance of 2.
         * @param familyName Name of AprilTag family
         * @return Augmented builder
         * @throws NullPointerException If family name is null
         * @throws IllegalArgumentException If family name is invalid
         */
        public Builder addFamily(String familyName) throws NullPointerException, IllegalArgumentException {
            return this.addFamily(familyName, AprilTagFamily.DEFAULT_HAMMING);
        }

        /**
         * Adds an AprilTag family to detect.
         * @param familyName Name of AprilTag family
         * @param bitsCorrected Maxumum number of bits to correct when detecting this family of AprilTags
         * @return
         * @throws NullPointerException If family name is null
         * @throws IllegalArgumentException If family name is invalid, or maxHamming is out of the range [0..3].
         */
        public Builder addFamily(String familyName, int bitsCorrected) throws NullPointerException, IllegalArgumentException {
            Objects.requireNonNull(familyName, "familyName");
            var family = AprilTagFamily.forName(familyName);
            return this.addFamily(family, bitsCorrected);
        }

        public Builder addFamily(AprilTagFamily family) throws IllegalArgumentException {
            return this.addFamily(family, AprilTagFamily.DEFAULT_HAMMING);
        }

        public Builder addFamily(AprilTagFamily family, int bitsCorrected) throws NullPointerException, IllegalArgumentException {
            Objects.requireNonNull(family, "family");
            family.validateBitsCorrected(bitsCorrected);
            this.families.put(family, bitsCorrected);
            return this;
        }

        public Builder removeFamily(AprilTagFamily family) {
            Objects.requireNonNull(family, "family");
            this.families.remove(family);
            return this;
        }

        public Builder clearFamilies() {
            this.families.clear();
            return this;
        }

        public AprilTagDetector build() {
            // We clone these values here
            var config = this.config.unmodifiable();
            var qtp = this.qtp.unmodifiable();
            var families = new LinkedHashMap<>(this.families);
            var familyPointerLookup = new HashMap<Long, AprilTagFamily>(this.families.size());

            // Acquire read locks for all AprilTag families
            List<ReentrantReadWriteLock.ReadLock> familyLocks = new ArrayList<>(this.families.size());
            try {
                // Reformat 
                int i = 0;
                long[] familyPtrs = new long[this.families.size()];
                int[] bitsCorrecteds = new int[this.families.size()];
                for (var entry : this.families.entrySet()) {
                    var family = Objects.requireNonNull(entry.getKey());
                    var bitsCorrected = Objects.requireNonNull(entry.getValue()).intValue();
                    family.validateBitsCorrected(bitsCorrected);
                    var lock = family.ptrLock.readLock();
                    lock.lock();
                    try {
                        familyPtrs[i] = family.ptr;
                        bitsCorrecteds[i] = bitsCorrected;
                        familyPointerLookup.put(family.ptr, family);
                        i++;
                        familyLocks.add(lock);
                    } catch (Throwable t) {
                        // Unlock pointer if we won't do this in the finally clause later
                        if (familyLocks.size() == 0 || familyLocks.get(familyLocks.size() - 1) != lock) {
                            lock.unlock();
                        }
                        throw t;
                    }
                }
                assert i == familyPtrs.length;

                long ptr = AprilTagDetector.nativeCreate(
                    config.numThreads,
                    config.quadDecimate,
                    config.quadSigma,
                    config.refineEdges,
                    config.decodeSharpening,
                    config.debug.toAbsolutePath().toString(),
                    qtp.minClusterPixels,
                    qtp.maxNumMaxima,
                    qtp.cosCriticalRad,
                    qtp.maxLineFitMSE,
                    qtp.minWhiteBlackDiff,
                    qtp.deglitch,
                    familyPtrs,
                    bitsCorrecteds
                );

                return new AprilTagDetector(ptr, config, qtp, families, familyPointerLookup);
            } finally {
                // Unlock the pointers for all the AprilTag families
                List<RuntimeException> exc = new ArrayList<>();
                try {
                    for (var lock : familyLocks) {
                        try {
                            lock.unlock();
                        } catch (RuntimeException e) {
                            exc.add(e);
                        }
                    }
                } finally {
                    if (!exc.isEmpty()) {
                        var last = exc.remove(exc.size() - 1);
                        for (var e : exc)
                            last.addSuppressed(e);
                        throw last;
                    }
                }
            }
        }
    }

    public static final class Accelerator {
        public static List<Accelerator> available() {
            throw new UnsupportedOperationException();
        }
    }

    /**
     * Configuration for AprilTagDetector
     */
    public static final class Config implements Cloneable {
        private final boolean mutable;
        private static int validateNumThreads(int numThreads) {
            if (numThreads < 0)
                throw new IllegalArgumentException("numThreads must be non-negative");
            return numThreads;
        }
        /**
         * How many threads should be used for computation.
         * 
         * Set to zero for automatic parallelism
         */
        private int numThreads = 1;
        private float quadDecimate = 2.0f;
        private float quadSigma = 0.0f;
        private boolean refineEdges = true;
        private float decodeSharpening = 0.25f;
        private Path debug = null;

        public Config() {
            this.mutable = true;
        }

        public Config(int numThreads, float quadDecimate, float quadSigma, boolean refineEdges, float decodeSharpening, boolean debug) {
            this(numThreads, quadDecimate, quadSigma, refineEdges, decodeSharpening, debug ? Path.of(".") : null);
        }

        public Config(int numThreads, float quadDecimate, float quadSigma, boolean refineEdges, float decodeSharpening, Path debug) {
            this.mutable = true;
            this.numThreads = validateNumThreads(numThreads);
            this.quadDecimate = quadDecimate;
            this.quadSigma = quadSigma;
            this.refineEdges = refineEdges;
            this.decodeSharpening = decodeSharpening;
            this.debug = debug;
        }

        public Config(Config config) {
            this(Objects.requireNonNull(config, "config"), true);
        }

        /**
         * Copy constructor
         */
        private Config(Config config, boolean mutable) {
            this.mutable = mutable;
            this.numThreads = validateNumThreads(config.numThreads);
            this.quadDecimate = config.quadDecimate;
            this.quadSigma = config.quadSigma;
            this.refineEdges = config.refineEdges;
            this.decodeSharpening = config.decodeSharpening;
            this.debug = config.debug;
        }

        /**
         * @return An umodifiable copy
         */
        public Config unmodifiable() {
            if (!this.isMutable())
                return this;
            return new Config(this, false);
        }

        /**
         * @return Whether this object can be mutated in-place
         */
        public boolean isMutable() {
            return mutable;
        }

        /**
         * Assert that this object is allowed to be mutated
         */
        private void assertMutable() throws UnsupportedOperationException {
            if (!this.mutable)
                throw new UnsupportedOperationException("Config is not mutable");
        }

        public int getNumThreads() {
            return this.numThreads;
        }

        public void setNumThreads(int numThreads) throws UnsupportedOperationException {
            this.assertMutable();
            this.numThreads = validateNumThreads(numThreads);
        }

        public float getQuadDecimate() {
            return this.quadDecimate;
        }

        public void setQuadDecimate(float quadDecimate) throws UnsupportedOperationException {
            this.assertMutable();
            this.quadDecimate = quadDecimate;
        }

        /**
         * What Gaussian blur should be applied to the segmented image
         * (used for quad detection?)  Parameter is the standard deviation
         * in pixels.  Very noisy images benefit from non-zero values
         * (e.g. 0.8).
         */
        public float getQuadSigma() {
            return this.quadSigma;
        }

        public void setQuadSigma(float quadSigma) throws UnsupportedOperationException {
            this.assertMutable();
            this.quadSigma = quadSigma;
        }

        public boolean getRefineEdges() {
            return this.refineEdges;
        }

        public void setRefineEdges(boolean refineEdges) throws UnsupportedOperationException {
            this.assertMutable();
            this.refineEdges = refineEdges;
        }

        public float getDecodeSharpening() {
            return this.decodeSharpening;
        }

        public void setDecodeSharpening(float decodeSharpening) throws UnsupportedOperationException {
            this.assertMutable();
            this.decodeSharpening = decodeSharpening;
        }

        public Path getDebug() {
            return this.debug;
        }

        public void setDebug(Path debug) throws UnsupportedOperationException {
            this.assertMutable();
            this.debug = debug;
        }

        @Override
        public Config clone() {
            return new Config(this, this.mutable);
        }

        @Override
        public String toString() {
            return "Config [nthreads=" + numThreads + ", quadDecimate=" + quadDecimate + ", quadSigma=" + quadSigma
                    + ", refineEdges=" + refineEdges + ", decodeSharpening=" + decodeSharpening + ", debug=" + debug
                    + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            //TODO: ignore mutable?
            result = prime * result + (mutable ? 1231 : 1237);
            result = prime * result + numThreads;
            result = prime * result + Float.floatToIntBits(quadDecimate);
            result = prime * result + Float.floatToIntBits(quadSigma);
            result = prime * result + (refineEdges ? 1231 : 1237);
            result = prime * result + Float.floatToIntBits(decodeSharpening);
            result = prime * result + Objects.hashCode(debug);
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null || !(obj instanceof Config))
                return false;
            Config other = (Config) obj;
            return (
                (this.mutable == other.mutable)
                && (this.numThreads == other.numThreads)
                && (Float.floatToIntBits(quadDecimate) == Float.floatToIntBits(other.quadDecimate))
                && (Float.floatToIntBits(quadSigma) == Float.floatToIntBits(other.quadSigma))
                && (refineEdges == other.refineEdges)
                && (Float.floatToIntBits(decodeSharpening) == Float.floatToIntBits(other.decodeSharpening))
                && Objects.equals(this.debug, other.debug)
            );
        }
    }

    public static final class QuadThresholdParameters implements Cloneable {
        private static int validateMinBlackWhiteDiff(int minWhiteBlackDiff) {
            if (minWhiteBlackDiff < 0 || 255 < minWhiteBlackDiff)
                throw new IllegalArgumentException("minWhiteBlackDiff must be in range 0..255");
            return minWhiteBlackDiff;
        }
        private final boolean mutable;
        /** Reject quads containing too few pixels */
        private int minClusterPixels = 5;

        /**
         * How many corner candidates to consider when segmenting a group
         * of pixels into a quad.
         */
        private int maxNumMaxima = 10;

        /**
         * Reject quads where pairs of edges have angles that are close to
         * straight or close to 180 degrees. Zero means that no quads are
         * rejected. (In radians).
         */
        private float cosCriticalRad = (float) Math.cos(10 * Math.PI / 180.0);
        /**
         * When fitting lines to the contours, what is the maximum mean
         * squared error allowed?
         * This is useful in rejecting contours that are far from being
         * quad shaped; rejecting these quads "early" saves expensive
         * decoding processing.
         */
        private float maxLineFitMSE = 10.0f;

        /**
         * When we build our model of black & white pixels, we add an
         * extra check that the white model must be (overall) brighter
         * than the black model. 
         * How much brighter? (in pixel values, [0,255]).
         */
        private int minWhiteBlackDiff = 5;

        /**
         * Should the thresholded image be deglitched?
         * Only useful for very noisy images
         */
        private boolean deglitch = false;

        public QuadThresholdParameters() {
            this.mutable = true;
        }

        public QuadThresholdParameters(int minClusterPixels, int maxNumMaxima, float cosCriticalRad, float maxLineFitMSE, int minWhiteBlackDiff, boolean deglitch) {
            this.minClusterPixels = minClusterPixels;
            this.maxNumMaxima = maxNumMaxima;
            this.cosCriticalRad = cosCriticalRad;
            this.maxLineFitMSE = maxLineFitMSE;
            this.minWhiteBlackDiff = validateMinBlackWhiteDiff(minWhiteBlackDiff);
            this.deglitch = deglitch;
            this.mutable = true;
        }

        public QuadThresholdParameters(QuadThresholdParameters source) {
            this(Objects.requireNonNull(source), true);
        }

        private QuadThresholdParameters(QuadThresholdParameters source, boolean mutable) {
            this.mutable = mutable;
            this.minClusterPixels = source.minClusterPixels;
            this.maxNumMaxima = source.maxNumMaxima;
            this.cosCriticalRad = source.cosCriticalRad;
            this.maxLineFitMSE = source.maxLineFitMSE;
            this.minWhiteBlackDiff = source.minWhiteBlackDiff;
            this.deglitch = source.deglitch;
        }

        /**
         * @return An umodifiable copy
         */
        public QuadThresholdParameters unmodifiable() {
            if (!this.isMutable())
                return this;
            return new QuadThresholdParameters(this, false);
        }

        /**
         * @return Whether this object can be mutated in-place
         */
        public boolean isMutable() {
            return mutable;
        }

        /**
         * Assert that this object is allowed to be mutated
         */
        private void assertMutable() throws UnsupportedOperationException {
            if (!this.mutable)
                throw new UnsupportedOperationException("Config is not mutable");
        }

        public int getMinClusterPixels() {
            return minClusterPixels;
        }

        public void setMinClusterPixels(int minClusterPixels) {
            this.assertMutable();
            this.minClusterPixels = minClusterPixels;
        }

        public int getMaxNumMaxima() {
            return maxNumMaxima;
        }

        public void setMaxNumMaxima(int maxNumMaxima) {
            this.assertMutable();
            this.maxNumMaxima = maxNumMaxima;
        }

        public float getCosCriticalRad() {
            return cosCriticalRad;
        }

        public void setCosCriticalRad(float cosCriticalRad) {
            this.assertMutable();
            this.cosCriticalRad = cosCriticalRad;
        }

        public float getMaxLineFitMSE() {
            return maxLineFitMSE;
        }

        public void setMaxLineFitMSE(float maxLineFitMSE) {
            this.assertMutable();
            this.maxLineFitMSE = maxLineFitMSE;
        }

        public int getMinWhiteBlackDiff() {
            return minWhiteBlackDiff;
        }

        public void setMinWhiteBlackDiff(int minWhiteBlackDiff) throws IllegalArgumentException {
            this.assertMutable();
            this.minWhiteBlackDiff = validateMinBlackWhiteDiff(minWhiteBlackDiff);
        }

        public boolean getDeglitch() {
            return deglitch;
        }

        public void setDeglitch(boolean deglitch) {
            this.assertMutable();
            this.deglitch = deglitch;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QuadThresholdParameters that = (QuadThresholdParameters) o;

            if (minClusterPixels != that.minClusterPixels) return false;
            if (maxNumMaxima != that.maxNumMaxima) return false;
            if (Float.compare(that.cosCriticalRad, cosCriticalRad) != 0) return false;
            if (Float.compare(that.maxLineFitMSE, maxLineFitMSE) != 0) return false;
            if (minWhiteBlackDiff != that.minWhiteBlackDiff) return false;
            return deglitch == that.deglitch;
        }

        @Override
        public int hashCode() {
            int result = minClusterPixels;
            result = 31 * result + maxNumMaxima;
            result = 31 * result + (cosCriticalRad != +0.0f ? Float.floatToIntBits(cosCriticalRad) : 0);
            result = 31 * result + (maxLineFitMSE != +0.0f ? Float.floatToIntBits(maxLineFitMSE) : 0);
            result = 31 * result + minWhiteBlackDiff;
            result = 31 * result + (deglitch ? 1 : 0);
            return result;
        }

        public QuadThresholdParameters clone() {
            return new QuadThresholdParameters(this, true);
        }
    }

    /** Configuration */
    private final Config config;
    /** QuadThresholdParameters */
    private final QuadThresholdParameters qtp;
    /** Map of family => hamming distance */
    private final Map<AprilTagFamily, Integer> families;
    /** Map of pointer => family */
    private final Map<Long, AprilTagFamily> ptrLookup;

    private AprilTagDetector(long ptr, Config config, QuadThresholdParameters qtp, Map<AprilTagFamily, Integer> families, Map<Long, AprilTagFamily> ptrLookup) {
        super(ptr);
        this.config = config;
        this.qtp = qtp;
        this.families = families;
        this.ptrLookup = ptrLookup;
    }

    public AprilTagDetections detect(byte[][] image) throws IllegalArgumentException, NullPointerException, NativeObjectReleasedException {
        var img = Image2D.from(image);
        return this.detect(img);
    }

    public AprilTagDetections detect(ByteBuffer buf, int width, int height) throws IllegalArgumentException, NullPointerException, NativeObjectReleasedException {
        return this.detect(buf, width, height, width);
    }

    /**
     * Detect AprilTags in image
     * @param buf Data buffer (8-bit grayscale format)
     * @param width Image width (pixels)
     * @param height Image height (pixels)
     * @param stride Image stride (bytes). Must be >= width.
     * @return Detections
     * @throws IllegalArgumentException If the image dimensions are invalid
     * @throws NativeObjectReleasedException If this method is called after the detector is closed
     * @throws NullPointerException If the buffer is null
     */
    public AprilTagDetections detect(ByteBuffer buf, int width, int height, int stride) throws IllegalArgumentException, NativeObjectReleasedException, NullPointerException {
        var image = Image2D.from(buf, width, height, stride);
        
        return this.detect(image);
    }

    public AprilTagDetections detect(Image2D image) {
        Image2D.validate(image);
        var width = image.getWidth();
        var height = image.getHeight();
        var stride = image.getStride();
        var buffer = image.buffer();

        return this.nativeRead(ptr -> AprilTagDetector.nativeDetect(ptr, buffer, width, height, stride, this.ptrLookup));
    }

    @Override
    protected native void destroy(long ptr);
}