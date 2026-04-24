/**
* This file is part of https://github.com/JingwenWang95/DSP-SLAM
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        std::cerr << std::endl << "Usage: ./dsp_slam_rgbd path_to_vocabulary path_to_settings path_to_sequence path_to_saved_trajectory" << std::endl;
        return 1;
    }

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    std::string strAssociationFilename = std::string(argv[3]) + "/associations.txt";
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    int nImages = vstrImageFilenamesRGB.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], argv[3], ORB_SLAM2::System::RGBD);

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl << std::endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni = 0; ni < nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(std::string(argv[3]) + "/" + vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(std::string(argv[3]) + "/" + vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            std::cerr << std::endl << "Failed to load image at: " << std::string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }

        if(imD.empty())
        {
            std::cerr << std::endl << "Failed to load depth image at: " << std::string(argv[3]) << "/" << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T = 0.0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<size_t>((T- ttrack)*1e6)));
        }

    }

    SLAM.SaveEntireMap(std::string(argv[4]));

    cv::waitKey(0);

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    std::sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    std::cout << "-------" << std::endl << std::endl;
    std::cout << "median tracking time: " << vTimesTrack[nImages/2] << std::endl;
    std::cout << "mean tracking time: " << totaltime/nImages << std::endl;

    return 0;
}

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps)
{
    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    if(!fAssociation.is_open())
    {
        std::cerr << "Failed to open associations file at: " << strAssociationFilename << std::endl;
        return;
    }

    while(!fAssociation.eof())
    {
        std::string s;
        std::getline(fAssociation,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double tRGB, tD;
            std::string sRGB, sD;
            ss >> tRGB;
            ss >> sRGB;
            ss >> tD;
            ss >> sD;

            vTimestamps.push_back(tRGB);
            vstrImageFilenamesRGB.push_back(sRGB);
            vstrImageFilenamesD.push_back(sD);
        }
    }
}
