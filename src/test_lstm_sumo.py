import os
import sys
import subprocess
import csv
from pathlib import Path

# --- 1. SETUP SUMO ENVIRONMENT ---
# Try to find SUMO_HOME, with fallbacks specifically for Mac Homebrew
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.path.append('/opt/homebrew/opt/sumo/share/sumo/tools')
    sys.path.append('/opt/homebrew/share/sumo/tools')

try:
    import traci
    from sumolib import checkBinary
except ImportError:
    print("Error: Could not import traci.")
    print("Run this in your terminal first: export SUMO_HOME='/opt/homebrew/opt/sumo/share/sumo'")
    sys.exit(1)


# --- 2. BUILD THE WORLD ---
def generate_network():
    """Use SUMO's netgenerate to build an instant 4-way intersection."""
    print("Building generic 4-way intersection (cross.net.xml)...")
    out_dir = Path("data/sumo")
    out_dir.mkdir(parents=True, exist_ok=True)
    net_file = out_dir / "cross.net.xml"
    
    # Updated to the correct SUMO 1.26 syntax
    cmd = [
        "netgenerate",
        "--grid",                     # Use the grid generator
        "--grid.x-number", "1",       # 1 intersection horizontally
        "--grid.y-number", "1",       # 1 intersection vertically
        "--grid.attach-length", "200",# Add 200-meter roads leading into the intersection
        "--default.lanenumber", "3",  # 3 lanes per road
        "--default.speed", "15",      # ~33 mph speed limit
        "--tls.guess",                # Guess and add Traffic Lights (TLS)
        "--output-file", str(net_file)
    ]
    subprocess.run(cmd, check=True)
    return str(net_file)

def generate_routes(route_file, num_trips=200):
    """Generate random traffic approaching the intersection."""
    print(f"Generating {num_trips} random vehicle trips (cross.rou.xml)...")
    net_file = Path("data/sumo/cross.net.xml")
    
    # Locate the randomTrips script
    sumo_home = os.environ.get('SUMO_HOME', '/opt/homebrew/opt/sumo/share/sumo')
    random_trips_path = os.path.join(sumo_home, "tools", "randomTrips.py")
    
    if not os.path.exists(random_trips_path):
        print("Warning: Could not find randomTrips.py. Writing a basic hardcoded route.")
        with open(route_file, "w") as routes:
            routes.write('<routes>\n')
            routes.write('    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="15"/>\n')
            routes.write('</routes>\n')
        return route_file

    cmd = [
        sys.executable, random_trips_path,
        "-n", str(net_file),
        "-r", str(route_file),
        "-e", str(num_trips),
        "--route-choice-method", "gawron"
    ]
    subprocess.run(cmd, check=True)
    return route_file


# --- 3. RUN THE SIMULATION & EXTRACT DATA ---
def run_simulation(net_file, route_file, output_csv="vehicle_data.csv"):
    """Start sumo-gui, run the simulation loop, and log data for the LSTM."""
    print("Starting SUMO-GUI...")
    sumoBinary = checkBinary('sumo-gui')
    
    traci.start([
        sumoBinary, 
        "-n", net_file, 
        "-r", route_file,
        "--step-length", "0.5" # Half-second increments
    ])
    
    print(f"Recording data to {output_csv}...")
    
    # Open a CSV file to store our training data
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['step', 'vehicle_id', 'x', 'y', 'speed_mps', 'lane_id', 'angle'])
        
        step = 0
        # Loop continues until all cars have left the map
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
            # 1. Get all cars currently on the map
            vehicle_ids = traci.vehicle.getIDList()
            
            for veh_id in vehicle_ids:
                # 2. Extract specific features for the LSTM
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                lane = traci.vehicle.getLaneID(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                
                # 3. Write this timestep's data to the CSV
                writer.writerow([step, veh_id, round(x, 2), round(y, 2), round(speed, 2), lane, round(angle, 2)])
            
            step += 1
            if step % 50 == 0:
                print(f"Simulation step {step}...")

    traci.close()
    sys.stdout.flush()
    print(f"Simulation complete! Successfully saved {step} steps of data to {output_csv}")


# --- 4. MAIN EXECUTION ---
def main():
    net_file = generate_network()
    
    route_file = Path("data/sumo") / "cross.rou.xml"
    route_file = generate_routes(str(route_file), num_trips=150) # Generates 150 cars
    
    run_simulation(net_file, route_file, output_csv="data/vehicle_data.csv")

if __name__ == "__main__":
    main()