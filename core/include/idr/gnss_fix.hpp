#pragma once

namespace idr {

struct GnssFix {
    double t;                 // seconds, monotonic
    double lat, lon, alt;     // WGS-84 coordinates (degrees, meters)
    double speed;             // m/s
    double accuracy;          // meters, 1-sigma or reported HDOP-derived
    int    sat_count;         // number of satellites
    bool   valid;             // validity flag
};

} // namespace idr
