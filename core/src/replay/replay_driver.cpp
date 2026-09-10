/**
 * @file replay_driver.cpp
 * @brief CLI Replay Driver for C++17 Strapdown Inertial Navigation System.
 *
 * Consumes pre-exported replay cache CSV files from data/processed/_cpp_replay_cache/
 * and runs real quaternion strapdown integration on raw IMU samples.
 * Generates 10 Hz trajectory predictions to data/processed/cpp_predictions/<outage_id>.csv.
 */

#include "idr/strapdown.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

void createDirectoryIfNotExists(const std::string& path) {
#if defined(_WIN32)
    CreateDirectoryA(path.c_str(), NULL);
#else
    mkdir(path.c_str(), 0755);
#endif
}

std::vector<std::string> listCsvFilesInDirectory(const std::string& dir) {
    std::vector<std::string> files;
#if defined(_WIN32)
    std::string search_pattern = dir + "\\*.csv";
    WIN32_FIND_DATAA find_data;
    HANDLE hFind = FindFirstFileA(search_pattern.c_str(), &find_data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.push_back(find_data.cFileName);
            }
        } while (FindNextFileA(hFind, &find_data) != 0);
        FindClose(hFind);
    }
#else
    DIR* dp = opendir(dir.c_str());
    if (dp != nullptr) {
        struct dirent* ep;
        while ((ep = readdir(dp)) != nullptr) {
            std::string name = ep->d_name;
            if (name.size() > 4 && name.substr(name.size() - 4) == ".csv") {
                files.push_back(name);
            }
        }
        closedir(dp);
    }
#endif
    return files;
}

struct InitialState {
    double t0{0.0};
    double lat0{0.0};
    double lon0{0.0};
    double alt0{0.0};
    double speed_ms{0.0};
    double heading_deg{0.0};
    double ax0{0.0};
    double ay0{0.0};
    double az0{idr::StrapdownIns::kGravity};
    bool valid{false};
};

InitialState parseHeaderComment(const std::string& line) {
    InitialState state;
    // Expected format:
    // # initial_state: t0=...,lat0=...,lon0=...,alt0=...,speed_ms=...,heading_deg=...,ax0=...,ay0=...,az0=...
    const std::string prefix = "# initial_state:";
    if (line.rfind(prefix, 0) != 0) {
        return state;
    }

    std::string params = line.substr(prefix.size());
    std::stringstream ss(params);
    std::string token;

    while (std::getline(ss, token, ',')) {
        size_t eq_pos = token.find('=');
        if (eq_pos == std::string::npos) continue;

        // Trim leading whitespace from key
        size_t key_start = token.find_first_not_of(" \t");
        if (key_start == std::string::npos) continue;
        std::string key = token.substr(key_start, eq_pos - key_start);
        std::string val_str = token.substr(eq_pos + 1);
        double val = std::strtod(val_str.c_str(), nullptr);

        if (key == "t0") state.t0 = val;
        else if (key == "lat0") state.lat0 = val;
        else if (key == "lon0") state.lon0 = val;
        else if (key == "alt0") state.alt0 = val;
        else if (key == "speed_ms") state.speed_ms = val;
        else if (key == "heading_deg") state.heading_deg = val;
        else if (key == "ax0") state.ax0 = val;
        else if (key == "ay0") state.ay0 = val;
        else if (key == "az0") state.az0 = val;
    }

    state.valid = true;
    return state;
}

struct RunAttitude {
    std::string run_id;
    bool has_motion{false};
    double yaw_deg{0.0};
    double pitch_rad{0.0};
    double roll_rad{0.0};
    std::string leveling_source;
    double gb_x{0.0};
    double gb_y{0.0};
    double gb_z{0.0};
    std::string gyro_bias_source;
    double qw{1.0};
    double qx{0.0};
    double qy{0.0};
    double qz{0.0};
    bool has_quaternion{false};
};

std::unordered_map<std::string, RunAttitude> loadAttitudeCsv(const std::string& path) {
    std::unordered_map<std::string, RunAttitude> attitudes;
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Warning: Could not open attitude CSV: " << path << "\n";
        return attitudes;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("run_id,", 0) == 0) continue; // skip header

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> cols;
        while (std::getline(ss, token, ',')) {
            cols.push_back(token);
        }
        if (cols.size() < 18) continue;

        RunAttitude att;
        att.run_id = cols[0];
        att.has_motion = (cols[1] == "True" || cols[1] == "true" || cols[1] == "1");
        if (!cols[3].empty()) att.yaw_deg = std::strtod(cols[3].c_str(), nullptr);
        if (!cols[5].empty()) att.pitch_rad = std::strtod(cols[5].c_str(), nullptr);
        if (!cols[6].empty()) att.roll_rad = std::strtod(cols[6].c_str(), nullptr);
        att.leveling_source = cols[9];
        if (!cols[10].empty()) att.gb_x = std::strtod(cols[10].c_str(), nullptr);
        if (!cols[11].empty()) att.gb_y = std::strtod(cols[11].c_str(), nullptr);
        if (!cols[12].empty()) att.gb_z = std::strtod(cols[12].c_str(), nullptr);
        att.gyro_bias_source = cols[13];

        if (!cols[14].empty() && !cols[15].empty() && !cols[16].empty() && !cols[17].empty()) {
            att.qw = std::strtod(cols[14].c_str(), nullptr);
            att.qx = std::strtod(cols[15].c_str(), nullptr);
            att.qy = std::strtod(cols[16].c_str(), nullptr);
            att.qz = std::strtod(cols[17].c_str(), nullptr);
            att.has_quaternion = true;
        }

        attitudes[att.run_id] = att;
    }
    return attitudes;
}

bool replayInstance(
    const std::string& input_csv_path,
    const std::string& output_csv_path,
    size_t& out_num_samples,
    const RunAttitude* att = nullptr
) {
    out_num_samples = 0;
    std::ifstream in(input_csv_path);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open input file: " << input_csv_path << "\n";
        return false;
    }

    // Read header line 1: Initial state
    std::string line1;
    if (!std::getline(in, line1)) {
        return false;
    }

    InitialState init_state = parseHeaderComment(line1);
    if (!init_state.valid) {
        std::cerr << "Error: Invalid header metadata in " << input_csv_path << "\n";
        return false;
    }

    // Read header line 2: Column names
    std::string line2;
    if (!std::getline(in, line2)) {
        return false;
    }

    // Initialize Strapdown INS
    idr::StrapdownIns ins;
    const idr::Vector3d initial_accel(init_state.ax0, init_state.ay0, init_state.az0);

    if (att != nullptr) {
        idr::Quaternion q0;
        if (att->has_quaternion) {
            // Full attitude quaternion from Module B (PCA yaw + Module A leveling, or PCA yaw + zero leveling)
            q0 = idr::Quaternion(att->qw, att->qx, att->qy, att->qz);
        } else if (att->leveling_source == "module_a_measured") {
            // Parked run (no vehicle motion, but measured static pitch, roll, and gyro bias)
            // Use coasting GNSS heading with measured pitch & roll
            q0 = idr::Quaternion::fromEulerEnu(init_state.heading_deg, att->pitch_rad, att->roll_rad);
        } else {
            // Fallback flat heading
            q0 = idr::Quaternion::fromHeadingEnu(init_state.heading_deg);
        }

        // Subtract gyro bias if from measured stationary window
        idr::Vector3d gyro_bias(0.0, 0.0, 0.0);
        if (att->gyro_bias_source == "module_a_measured") {
            gyro_bias = idr::Vector3d(att->gb_x, att->gb_y, att->gb_z);
        }

        ins.initializeWithAttitude(
            init_state.t0,
            init_state.lat0,
            init_state.lon0,
            init_state.alt0,
            init_state.speed_ms,
            init_state.heading_deg,
            q0,
            gyro_bias,
            initial_accel
        );
    } else {
        ins.initialize(
            init_state.t0,
            init_state.lat0,
            init_state.lon0,
            init_state.alt0,
            init_state.speed_ms,
            init_state.heading_deg,
            initial_accel
        );
    }

    std::ofstream out(output_csv_path);
    if (!out.is_open()) {
        std::cerr << "Error: Failed to open output file: " << output_csv_path << "\n";
        return false;
    }

    // Write prediction CSV header (matches Step 8 baseline schema)
    out << "timestamp_s,latitude_deg,longitude_deg,speed_ms,heading_deg\n";
    out << std::fixed;

    std::string row_line;
    while (std::getline(in, row_line)) {
        if (row_line.empty() || row_line[0] == '#') continue;

        std::stringstream ss(row_line);
        std::string t_str, ax_str, ay_str, az_str, gx_str, gy_str, gz_str, mx_str, my_str, mz_str;

        if (!std::getline(ss, t_str, ',') ||
            !std::getline(ss, ax_str, ',') ||
            !std::getline(ss, ay_str, ',') ||
            !std::getline(ss, az_str, ',') ||
            !std::getline(ss, gx_str, ',') ||
            !std::getline(ss, gy_str, ',') ||
            !std::getline(ss, gz_str, ',')) {
            continue;
        }

        // Optional magnetometer
        std::getline(ss, mx_str, ',');
        std::getline(ss, my_str, ',');
        std::getline(ss, mz_str, ',');

        idr::ImuSample sample;
        sample.t = std::strtod(t_str.c_str(), nullptr);
        sample.ax = std::strtod(ax_str.c_str(), nullptr);
        sample.ay = std::strtod(ay_str.c_str(), nullptr);
        sample.az = std::strtod(az_str.c_str(), nullptr);
        sample.gx = std::strtod(gx_str.c_str(), nullptr);
        sample.gy = std::strtod(gy_str.c_str(), nullptr);
        sample.gz = std::strtod(gz_str.c_str(), nullptr);
        sample.mx = mx_str.empty() ? 0.0 : std::strtod(mx_str.c_str(), nullptr);
        sample.my = my_str.empty() ? 0.0 : std::strtod(my_str.c_str(), nullptr);
        sample.mz = mz_str.empty() ? 0.0 : std::strtod(mz_str.c_str(), nullptr);

        ins.update(sample);

        out << std::setprecision(6) << sample.t << ","
            << std::setprecision(8) << ins.getLatitude() << ","
            << std::setprecision(8) << ins.getLongitude() << ","
            << std::setprecision(4) << ins.getSpeed() << ","
            << std::setprecision(4) << ins.getHeadingDeg() << "\n";

        out_num_samples++;
    }

    return true;
}
std::string extractRunId(const std::string& filename) {
    std::string name = filename;
    if (name.size() >= 4 && name.substr(name.size() - 4) == ".csv") {
        name = name.substr(0, name.size() - 4);
    }
    size_t pos = name.find("__");
    if (pos != std::string::npos) {
        return name.substr(0, pos);
    }
    return name;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string cache_dir = "data/processed/_cpp_replay_cache";
    std::string out_dir = "data/processed/cpp_predictions";
    std::string attitude_csv = "";
    std::string single_outage_id = "";
    bool batch_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--cache-dir" || arg == "-c") && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if ((arg == "--out-dir" || arg == "-o") && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--attitude-csv" && i + 1 < argc) {
            attitude_csv = argv[++i];
        } else if ((arg == "--outage-id" || arg == "-i") && i + 1 < argc) {
            single_outage_id = argv[++i];
        } else if (arg == "--batch" || arg == "-b") {
            batch_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: idr_replay [options]\n"
                      << "Options:\n"
                      << "  --cache-dir <dir>     Directory containing replay cache CSVs (default: data/processed/_cpp_replay_cache)\n"
                      << "  --out-dir <dir>       Destination directory for prediction CSVs (default: data/processed/cpp_predictions)\n"
                      << "  --attitude-csv <path> Path to module_b_initial_attitude.csv for initial attitude & gyro bias\n"
                      << "  --batch               Run across all cached outage instances\n"
                      << "  --outage-id <id>      Replay a specific outage instance by ID\n"
                      << "  --help, -h            Show this help message\n";
            return 0;
        }
    }

    createDirectoryIfNotExists(out_dir);

    std::unordered_map<std::string, RunAttitude> attitudes;
    if (!attitude_csv.empty()) {
        attitudes = loadAttitudeCsv(attitude_csv);
        std::cout << "Loaded " << attitudes.size() << " initial attitudes from " << attitude_csv << "\n";
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (!single_outage_id.empty()) {
        std::string filename = single_outage_id;
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".csv") {
            filename += ".csv";
        }
        std::string in_path = cache_dir + "/" + filename;
        std::string out_path = out_dir + "/" + filename;

        const RunAttitude* att_ptr = nullptr;
        if (!attitudes.empty()) {
            std::string run_id = extractRunId(filename);
            auto it = attitudes.find(run_id);
            if (it != attitudes.end()) {
                att_ptr = &it->second;
            } else {
                std::cout << "Notice: run_id '" << run_id << "' not found in attitude CSV, using default.\n";
            }
        }

        std::cout << "Replaying single outage: " << filename << "\n";
        size_t num_samples = 0;
        if (!replayInstance(in_path, out_path, num_samples, att_ptr)) {
            std::cerr << "Failed to replay " << filename << "\n";
            return 1;
        }
        std::cout << "  Generated " << num_samples << " trajectory points -> " << out_path << "\n";
        return 0;
    }

    if (!batch_mode) {
        batch_mode = true;
    }

    std::vector<std::string> files = listCsvFilesInDirectory(cache_dir);
    if (files.empty()) {
        std::cerr << "Error: No CSV files found in cache directory: " << cache_dir << "\n";
        return 1;
    }

    std::cout << "Starting C++ Strapdown INS Batch Replay:\n";
    std::cout << "  Cache Directory:  " << cache_dir << "\n";
    std::cout << "  Output Directory: " << out_dir << "\n";
    std::cout << "  Attitude CSV:     " << (attitude_csv.empty() ? "(none)" : attitude_csv) << "\n";
    std::cout << "  Total Files:      " << files.size() << "\n\n";

    size_t processed_count = 0;
    size_t skipped_count = 0;
    size_t error_count = 0;
    size_t total_points = 0;

    for (const auto& file : files) {
        const RunAttitude* att_ptr = nullptr;
        if (!attitude_csv.empty()) {
            std::string run_id = extractRunId(file);
            auto it = attitudes.find(run_id);
            if (it == attitudes.end()) {
                skipped_count++;
                continue;
            }
            att_ptr = &it->second;
        }

        std::string in_path = cache_dir + "/" + file;
        std::string out_path = out_dir + "/" + file;

        size_t samples = 0;
        if (replayInstance(in_path, out_path, samples, att_ptr)) {
            processed_count++;
            total_points += samples;
        } else {
            error_count++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "\n========================================\n";
    std::cout << " [COMPLETED] Batch Replay Summary:\n";
    std::cout << "  Successfully Processed: " << processed_count << " instances\n";
    if (skipped_count > 0) {
        std::cout << "  Skipped (Non-Eligible): " << skipped_count << " instances\n";
    }
    std::cout << "  Failed Instances:       " << error_count << "\n";
    std::cout << "  Total Trajectory Ticks: " << total_points << " points\n";
    std::cout << "  Total Replay Time:      " << duration_ms << " ms ("
              << (duration_ms / (processed_count ? processed_count : 1)) << " ms/instance)\n";
    std::cout << "========================================\n";

    return (error_count == 0) ? 0 : 1;
}
