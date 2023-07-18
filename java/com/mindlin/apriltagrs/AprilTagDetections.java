package com.mindlin.apriltagrs;

import java.util.AbstractList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.RandomAccess;
import com.mindlin.apriltagrs.debug.TimeProfile;

public final class AprilTagDetections extends AbstractList<AprilTagDetection> implements RandomAccess {
    private int numQuads;
    private TimeProfile tp;
    private List<AprilTagDetection> detections;

    public int getNumQuads() {
        return this.numQuads;
    }

    public Optional<TimeProfile> getTimeProfile() {
        return Optional.ofNullable(this.tp);
    }
    
    @Override
    public int size() {
        return this.detections.size();
    }

    @Override
    public AprilTagDetection get(int index) {
        return this.detections.get(index);
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
}