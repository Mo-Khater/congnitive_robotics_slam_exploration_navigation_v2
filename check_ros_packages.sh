#!/bin/bash

##############################################################################
# ROS Package Installation Checker for LIMO Exploration Project
# This script verifies all required ROS packages are installed
##############################################################################

echo "========================================================================"
echo "  ROS Package Installation Checker for LIMO Robot Exploration"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for missing packages
MISSING_COUNT=0
INSTALLED_COUNT=0

# Function to check if a package exists
check_package() {
    local pkg_name=$1
    local description=$2
    
    if rospack find "$pkg_name" &> /dev/null; then
        echo -e "${GREEN}[✓]${NC} $pkg_name - $description"
        ((INSTALLED_COUNT++))
    else
        echo -e "${RED}[✗]${NC} $pkg_name - $description ${RED}(MISSING)${NC}"
        ((MISSING_COUNT++))
    fi
}

echo "Checking required ROS packages..."
echo "------------------------------------------------------------------------"

# Core ROS packages (usually pre-installed)
echo ""
echo "Core ROS Packages:"
check_package "tf2_ros" "Transform library"
check_package "robot_state_publisher" "Publishes robot state to TF"
check_package "rviz" "Visualization tool"
check_package "xacro" "XML macro processor for URDF"

# LIMO-specific packages
echo ""
echo "LIMO Robot Packages:"
check_package "limo_bringup" "LIMO hardware interface"
check_package "limo_description" "LIMO robot URDF models"

# SLAM package
echo ""
echo "SLAM Packages:"
check_package "gmapping" "GMapping SLAM algorithm"

# Navigation packages
echo ""
echo "Navigation Packages:"
check_package "move_base" "Navigation framework"
check_package "map_server" "Map loading/saving"
check_package "amcl" "Localization (optional for exploration)"

# Exploration package
echo ""
echo "Exploration Packages:"
check_package "explore_lite" "Autonomous exploration"

# Your custom package
echo ""
echo "Custom Package:"
check_package "limo_gazebo_sim" "Your custom project package"

echo ""
echo "========================================================================"
echo "Summary:"
echo "------------------------------------------------------------------------"
echo -e "Installed packages: ${GREEN}$INSTALLED_COUNT${NC}"
echo -e "Missing packages:   ${RED}$MISSING_COUNT${NC}"
echo "========================================================================"

# Provide installation commands if packages are missing
if [ $MISSING_COUNT -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}INSTALLATION COMMANDS:${NC}"
    echo "------------------------------------------------------------------------"
    echo ""
    
    # Check which packages are missing and provide install commands
    if ! rospack find gmapping &> /dev/null; then
        echo "# Install GMapping:"
        echo "sudo apt-get install ros-\$ROS_DISTRO-gmapping"
        echo ""
    fi
    
    if ! rospack find move_base &> /dev/null; then
        echo "# Install Navigation Stack:"
        echo "sudo apt-get install ros-\$ROS_DISTRO-navigation"
        echo ""
    fi
    
    if ! rospack find explore_lite &> /dev/null; then
        echo "# Install Explore Lite:"
        echo "sudo apt-get install ros-\$ROS_DISTRO-explore-lite"
        echo ""
        echo "# OR build from source if not available:"
        echo "cd ~/catkin_ws/src"
        echo "git clone https://github.com/hrnr/m-explore.git"
        echo "cd ~/catkin_ws"
        echo "catkin_make"
        echo ""
    fi
    
    if ! rospack find limo_bringup &> /dev/null || ! rospack find limo_description &> /dev/null; then
        echo "# Install LIMO packages (usually provided by manufacturer):"
        echo "cd ~/catkin_ws/src"
        echo "# Clone LIMO ROS package from manufacturer"
        echo "# git clone <LIMO_ROS_PACKAGE_URL>"
        echo "cd ~/catkin_ws"
        echo "catkin_make"
        echo ""
    fi
    
    if ! rospack find limo_gazebo_sim &> /dev/null; then
        echo "# Create your custom limo_gazebo_sim package:"
        echo "cd ~/catkin_ws/src"
        echo "catkin_create_pkg limo_gazebo_sim rospy std_msgs sensor_msgs geometry_msgs nav_msgs"
        echo "cd ~/catkin_ws"
        echo "catkin_make"
        echo ""
    fi
    
    echo "After installing missing packages, run:"
    echo "source ~/catkin_ws/devel/setup.bash"
    echo ""
    
    exit 1
else
    echo ""
    echo -e "${GREEN}✓ All required packages are installed!${NC}"
    echo ""
    exit 0
fi