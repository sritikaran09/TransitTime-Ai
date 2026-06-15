import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function App() {
  const [inputs, setInputs] = useState({
    source_lat: "20.2382002197894",
    source_lon: "85.8320922681074",
    destination_lat: "20.395",
    destination_lon: "85.826",
    date_str: new Date().toISOString().split('T')[0],
    selected_hour: "14" // New state key for the specific hour calculation
  });

  const [routeInfo, setRouteInfo] = useState(null); // Stores the geocoded address payloads
  const [graphData, setGraphData] = useState([]);
  const [singlePrediction, setSinglePrediction] = useState(null); // Stores specific hour data
  const [loadingTimeline, setLoadingTimeline] = useState(false);
  const [loadingSingle, setLoadingSingle] = useState(false);

  // Payload body helper to keep code DRY
  const getPayloadBody = () => ({
    source_lat: parseFloat(inputs.source_lat),
    source_lon: parseFloat(inputs.source_lon),
    destination_lat: parseFloat(inputs.destination_lat),
    destination_lon: parseFloat(inputs.destination_lon),
    date_str: inputs.date_str
  });

  // Action 1: Fetch 24-Hour Profile Timeline Chart Array
  const handleFetchTimeline = async () => {
    setLoadingTimeline(true);
    setRouteInfo(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict-timeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(getPayloadBody())
      });
      const data = await response.json();
      
      setGraphData(data.timeline);
      // Capture the backend geocoded locations summary to show on screen
      setRouteInfo({
        from: data.from_location,
        to: data.to_location
      });
    } catch (error) {
      console.error("Error communicating with ML backend:", error);
    }
    setLoadingTimeline(false);
  };

  // Action 2: Fetch Specific Single Hour Response Data
  const handleFetchSingleHour = async () => {
    setLoadingSingle(true);
    setSinglePrediction(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict-single-hour', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...getPayloadBody(),
          selected_hour: parseInt(inputs.selected_hour)
        })
      });
      const data = await response.json();
      setSinglePrediction(data);
      
      // Update route info addresses simultaneously if not already loaded
      if (!routeInfo) {
        setRouteInfo({
          from: data.from_location,
          to: data.to_location
        });
      }
    } catch (error) {
      console.error("Error evaluating single snapshot:", error);
    }
    setLoadingSingle(false);
  };

  return (
    <div style={{ padding: '30px', fontFamily: 'Segoe UI, sans-serif', maxWidth: '1000px', margin: '0 auto', color: '#333' }}>
      <h2 style={{ borderBottom: '2px solid #eee', paddingBottom: '10px', color: '#0056b3' }}>
        Transit Intelligence Portal
      </h2>
      
      {/* Parameters Input Panel Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', margin: '20px 0', background: '#f8f9fa', padding: '20px', borderRadius: '8px', border: '1px solid #e9ecef' }}>
        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>Source Coordinates (Lat / Lon)</label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input type="text" value={inputs.source_lat} placeholder="Latitude" onChange={e => setInputs({...inputs, source_lat: e.target.value})} style={{ flex: 1, padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }} />
            <input type="text" value={inputs.source_lon} placeholder="Longitude" onChange={e => setInputs({...inputs, source_lon: e.target.value})} style={{ flex: 1, padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }} />
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>Destination Coordinates (Lat / Lon)</label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input type="text" value={inputs.destination_lat} placeholder="Latitude" onChange={e => setInputs({...inputs, destination_lat: e.target.value})} style={{ flex: 1, padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }} />
            <input type="text" value={inputs.destination_lon} placeholder="Longitude" onChange={e => setInputs({...inputs, destination_lon: e.target.value})} style={{ flex: 1, padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }} />
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>Target Run Date</label>
          <input type="date" value={inputs.date_str} onChange={e => setInputs({...inputs, date_str: e.target.value})} style={{ padding: '8px', width: '100%', boxSizing: 'border-box', borderRadius: '4px', border: '1px solid #ccc' }} />
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>Specific Dispatch Hour (0-23)</label>
          <input type="number" min="0" max="23" value={inputs.selected_hour} onChange={e => setInputs({...inputs, selected_hour: e.target.value})} style={{ padding: '8px', width: '100%', boxSizing: 'border-box', borderRadius: '4px', border: '1px solid #ccc' }} />
        </div>

        <div style={{ gridColumn: 'span 2', display: 'flex', gap: '15px', marginTop: '10px' }}>
          <button onClick={handleFetchTimeline} style={{ padding: '10px 20px', background: '#28a745', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', flex: 1 }}>
            {loadingTimeline ? 'Mapping Timeline Array...' : 'Generate 24h Profile Chart'}
          </button>
          <button onClick={handleFetchSingleHour} style={{ padding: '10px 20px', background: '#007bff', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', flex: 1 }}>
            {loadingSingle ? 'Calculating Snapshot...' : 'Predict Specific Hour Only'}
          </button>
        </div>
      </div>

      {/* Geocoded Address Info Card */}
      {routeInfo && (
        <div style={{ margin: '20px 0', padding: '15px', background: '#e9f5ff', borderLeft: '5px solid #007bff', borderRadius: '4px' }}>
          <h4 style={{ margin: '0 0 10px 0', color: '#004085' }}>Resolved Network Geocoding Routing Paths</h4>
          <p style={{ margin: '5px 0' }}><strong>From Address:</strong> {routeInfo.from.full_address} <span style={{ color: '#666', fontSize: '0.9em' }}>({routeInfo.from.short_summary})</span></p>
          <p style={{ margin: '5px 0' }}><strong>To Address:</strong> {routeInfo.to.full_address} <span style={{ color: '#666', fontSize: '0.9em' }}>({routeInfo.to.short_summary})</span></p>
        </div>
      )}

      {/* Specific Hour Output Result Box */}
      {singlePrediction && (
        <div style={{ margin: '20px 0', padding: '20px', background: '#fff3cd', border: '1px solid #ffeeba', borderRadius: '8px', textAlign: 'center' }}>
          <span style={{ fontSize: '1.2em', fontWeight: '500', color: '#856404' }}>
            Predicted Transit Time at <strong>{singlePrediction.departure_time}</strong> departure slot:
          </span>
          <div style={{ fontSize: '2.5em', fontWeight: 'bold', color: '#856404', marginTop: '10px' }}>
            {singlePrediction.predicted_transit_time}
          </div>
        </div>
      )}

      {/* Responsive Visual Timeline Chart Canvas */}
      {graphData.length > 0 ? (
        <div style={{ marginTop: '20px', padding: '20px', background: '#fdfdfd', border: '1px solid #ddd', borderRadius: '8px' }}>
          <h3 style={{ margin: '0 0 15px 0' }}>Hourly Transit Variance Profile Map</h3>
          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer>
              <LineChart data={graphData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis label={{ value: 'Duration (Min)', angle: -90, position: 'insideLeft', offset: 15 }} />
                <Tooltip formatter={(value) => [`${value} Minutes`, 'Transit Duration Estimate']} />
                <Line type="monotone" dataKey="transit_time" stroke="#28a745" strokeWidth={3} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        !loadingTimeline && !singlePrediction && (
          <div style={{ textAlign: 'center', color: '#777', padding: '50px', background: '#f9f9f9', borderRadius: '8px', border: '1px dashed #ccc' }}>
            Submit route coordinates above to unpack your ensemble model estimations.
          </div>
        )
      )}
    </div>
  );
}