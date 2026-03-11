import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import api from '../lib/api';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Calendar, MapPin, Clock, Phone, CheckCircle, XCircle, CircleDashed, Loader2 } from 'lucide-react';

// Fix Leaflet icon
import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Compact numbered marker
const createNumberedIcon = (number, status) => {
  let bgColor = '#f59e0b'; // Planned/amber
  if (status === 'Done' || status === 'Completed') bgColor = '#10b981'; // green
  if (status === 'Cancelled') bgColor = '#ef4444'; // red
  return L.divIcon({
    className: 'custom-marker',
    html: `<div class="w-8 h-8 rounded-full bg-white shadow-lg border-2 flex items-center justify-center text-white font-bold text-sm" style="background-color:${bgColor}">${number}</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
};

// Helper for local date string YYYY-MM-DD
const getLocalDateString = () => {
  const d = new Date();
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

export default function Dashboard() {
  const { user, logout } = useAuth();
  const isAdmin = user?.role?.toLowerCase() === 'admin' || user?.user_id?.toLowerCase() === 'admin';

  const [date, setDate] = useState(getLocalDateString());
  const [selectedMrId, setSelectedMrId] = useState(''); // empty = current user or all
  const [mrs, setMrs] = useState([]); // MR list for admin dropdown
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTask, setSelectedTask] = useState(null); // For modal popup

  // Fetch MR list (only for admin)
  useEffect(() => {
    if (isAdmin) {
      api.get('/admin/mrs')
        .then(res => {
          console.log('MR list loaded:', res.data);
          setMrs(res.data || []);
          if (res.data && res.data.length > 0) {
            setSelectedMrId(res.data[0].mr_id);
          }
        })
        .catch(err => {
          console.error('Failed to load MRs:', err);
          setError('Failed to load MR list');
        });
    }
  }, [isAdmin]);

  // Fetch tasks when date or MR selection changes
  useEffect(() => {
    if (!user?.user_id) {
      setError("User ID missing. Please log in again.");
      setLoading(false);
      return;
    }

    const safeDate = new Date(date).toISOString().split('T')[0];
    let mrIdToUse = user.user_id;

    // Admin can override with selected MR
    if (isAdmin && selectedMrId) {
      mrIdToUse = selectedMrId;
    } else if (isAdmin && !selectedMrId) {
      // If admin and no MR selected, do not fetch yet waiting for auto-select
      setLoading(false);
      return;
    }

    const endpoint = `/schedule/daily/${encodeURIComponent(mrIdToUse)}/${safeDate}`;

    setLoading(true);
    setError(null);

    api
      .get(endpoint)
      .then(response => setTasks(response.data || []))
      .catch(err => setError(err.response?.data?.detail || 'Failed to load schedule'))
      .finally(() => setLoading(false));
  }, [date, user?.user_id, isAdmin, selectedMrId]);

  const updateTaskStatus = async (taskId, newStatus) => {
    try {
      await api.put('/schedule/status', { activity_id: taskId, status: newStatus });
      setTasks(prev => prev.map(t => t.activity_id === taskId ? { ...t, status: newStatus } : t));
    } catch (err) {
      console.error('Status update failed:', err);
      setError('Failed to update task status');
    }
  };

  // Map data
  const validTasks = tasks
    .filter(t => t.latitude && t.longitude)
    .map((t, index) => ({
      ...t,
      number: index + 1,
      position: [parseFloat(t.latitude), parseFloat(t.longitude)],
    }));

  const positions = validTasks.map(t => t.position);
  const center = positions.length > 0 ? positions[0] : [23.0225, 72.5714];

  // Categorize tasks
  const plannedTasks = tasks.filter(t => !t.status || t.status === 'Planned' || t.status === 'Pending');
  const completedTasks = tasks.filter(t => t.status === 'Done' || t.status === 'Completed');
  const cancelledTasks = tasks.filter(t => t.status === 'Cancelled');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gradient-to-br dark:from-zinc-900 dark:to-zinc-950 p-6">
      <div className="w-full space-y-8">
        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">
              Overview
            </h1>
            <p className="text-zinc-900 dark:text-zinc-300 mt-1 flex items-center gap-2 font-medium">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              {user?.name || user?.user_id}
            </p>
          </div>
          <button
            onClick={logout}
            className="px-6 py-2.5 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-lg hover:from-red-500 hover:to-red-400 transition-all shadow-md text-sm font-bold flex items-center gap-2"
          >
            Sign Out
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 bg-white dark:bg-zinc-900 p-4 rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm">
          <div className="flex items-center gap-4">
            <button
              onClick={() => {
                const newDate = new Date(date);
                newDate.setDate(newDate.getDate() - 1);
                setDate(newDate.toISOString().split('T')[0]);
              }}
              className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition"
            >
              <ChevronLeft size={20} className="text-zinc-600 dark:text-zinc-400" />
            </button>

            <span
              className="font-black text-xl text-black dark:text-white min-w-[160px] text-center cursor-pointer hover:text-zinc-600 transition"
              onClick={() => document.getElementById('date-input').showPicker()}
            >
              {new Date(date).toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' })}
            </span>

            <input
              id="date-input"
              type="date"
              value={date}
              onChange={e => {
                if (e.target.value) setDate(e.target.value);
              }}
              className="absolute opacity-0 w-0 h-0 pointer-events-none"
            />

            <button
              onClick={() => {
                const newDate = new Date(date);
                newDate.setDate(newDate.getDate() + 1);
                setDate(newDate.toISOString().split('T')[0]);
              }}
              className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-lg transition"
            >
              <ChevronRight size={20} className="text-zinc-600 dark:text-zinc-400" />
            </button>
          </div>

          {isAdmin && (
            <div className="w-full md:w-auto">
              <select
                value={selectedMrId}
                onChange={e => setSelectedMrId(e.target.value)}
                className="w-full md:w-64 px-4 py-2 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg text-sm text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-zinc-500 outline-none font-bold"
              >
                {mrs.map(mr => (
                  <option key={mr.mr_id} value={mr.mr_id}>
                    {mr.display_name || mr.mr_id} ({mr.mr_id})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {loading ? (
          <div className="text-center py-20">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-zinc-800 dark:border-zinc-200 mx-auto"></div>
          </div>
        ) : error ? (
          <div className="bg-red-50 dark:bg-red-900/10 p-4 rounded-lg text-red-600 dark:text-red-400 border border-red-100 dark:border-red-900/30 text-center text-sm">
            {error}
          </div>
        ) : (
          <>
            {/* KPI Cards */}
            <div className="grid grid-cols-3 gap-6">
              {[
                { label: 'Total Visits', value: tasks.length, color: 'text-black dark:text-white' },
                { label: 'Completed', value: completedTasks.length, color: 'text-green-700 dark:text-green-400' },
                { label: 'Cancelled', value: cancelledTasks.length, color: 'text-red-700 dark:text-red-400' }
              ].map((kpi, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl p-6 rounded-xl border border-zinc-200/50 dark:border-white/10 shadow-sm relative overflow-hidden group"
                >
                  <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-white/5 to-transparent rounded-full -mr-8 -mt-8 pointer-events-none group-hover:scale-110 transition-transform duration-500" />
                  <p className="text-xs uppercase tracking-wider text-black dark:text-zinc-300 font-extrabold relative z-10">{kpi.label}</p>
                  <p className={`text-5xl font-black mt-2 ${kpi.color} relative z-10`}>{kpi.value}</p>
                </motion.div>
              ))}
            </div>

            {/* Map */}
            <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl rounded-xl border border-zinc-200/50 dark:border-white/10 shadow-sm overflow-hidden h-[400px]">
              <MapContainer center={center} zoom={12} style={{ height: '100%', width: '100%' }}>
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                {validTasks.map(task => (
                  <Marker key={task.activity_id} position={task.position} icon={createNumberedIcon(task.number, task.status)}>
                    <Popup>{task.customer_name}</Popup>
                  </Marker>
                ))}
                {positions.length > 1 && <Polyline positions={positions} color="#3f3f46" weight={3} opacity={0.6} />}
              </MapContainer>
            </div>

            {/* Tasks Columns */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Planned */}
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-zinc-200 dark:border-zinc-800">
                  <h3 className="font-extrabold text-black dark:text-white flex items-center gap-2 text-lg">
                    <div className="w-2.5 h-2.5 rounded-full bg-amber-500"></div> Planned
                  </h3>
                  <span className="text-sm font-black text-zinc-900 dark:text-zinc-100">{plannedTasks.length}</span>
                </div>
                {plannedTasks.map((task) => (
                  <TaskCard
                    key={task.activity_id}
                    task={task}
                    status="Planned"
                    onUpdate={updateTaskStatus}
                    onClick={setSelectedTask}
                    isEditable={date === getLocalDateString()}
                  />
                ))}
                {plannedTasks.length === 0 && <EmptyState />}
              </div>

              {/* Completed */}
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-zinc-200 dark:border-zinc-800">
                  <h3 className="font-extrabold text-black dark:text-white flex items-center gap-2 text-lg">
                    <div className="w-2.5 h-2.5 rounded-full bg-green-600"></div> Completed
                  </h3>
                  <span className="text-sm font-black text-zinc-900 dark:text-zinc-100">{completedTasks.length}</span>
                </div>
                {completedTasks.map((task) => (
                  <TaskCard
                    key={task.activity_id}
                    task={task}
                    status="Completed"
                    onUpdate={updateTaskStatus}
                    isEditable={date === getLocalDateString()}
                  />
                ))}
                {completedTasks.length === 0 && <EmptyState />}
              </div>

              {/* Cancelled */}
              <div className="space-y-4">
                <div className="flex items-center justify-between pb-2 border-b border-zinc-200 dark:border-zinc-800">
                  <h3 className="font-extrabold text-black dark:text-white flex items-center gap-2 text-lg">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-600"></div> Cancelled
                  </h3>
                  <span className="text-sm font-black text-zinc-900 dark:text-zinc-100">{cancelledTasks.length}</span>
                </div>
                {cancelledTasks.map((task) => (
                  <TaskCard
                    key={task.activity_id}
                    task={task}
                    status="Cancelled"
                    onUpdate={updateTaskStatus}
                    isEditable={date === getLocalDateString()}
                  />
                ))}
                {cancelledTasks.length === 0 && <EmptyState />}
              </div>
            </div>

            {/* Modal */}
            <TaskDetailsModal task={selectedTask} onClose={() => setSelectedTask(null)} />
          </>
        )}
      </div>
    </div>
  );
}

// Subcomponents for cleaner code
function EmptyState() {
  return (
    <div className="py-8 text-center border border-dashed border-zinc-300/50 dark:border-white/10 rounded-xl bg-zinc-50/30 dark:bg-zinc-900/30 backdrop-blur-sm">
      <p className="text-xs text-zinc-500 dark:text-zinc-400 font-semibold uppercase tracking-wider">No activities</p>
    </div>
  )
}

function TaskDetailsModal({ task, onClose }) {
  if (!task) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm" onClick={onClose}>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        onClick={e => e.stopPropagation()}
        className="bg-white dark:bg-zinc-900 rounded-2xl shadow-xl w-full max-w-md overflow-hidden border border-zinc-200 dark:border-white/10"
      >
        <div className="p-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="text-xl font-bold text-zinc-900 dark:text-white">{task.customer_name}</h3>
              <p className="text-sm text-zinc-900 dark:text-zinc-300 font-medium">Activity Details</p>
            </div>
            <button onClick={onClose} className="p-1 hover:bg-zinc-100 dark:hover:bg-white/10 rounded-full transition-colors text-zinc-500">
              <XCircle size={20} />
            </button>
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-3 p-3 bg-zinc-50 dark:bg-white/5 rounded-lg">
              <div className="p-2 bg-indigo-100 dark:bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 rounded-lg">
                <Phone size={18} />
              </div>
              <div>
                <p className="text-xs text-zinc-900 dark:text-zinc-300 uppercase tracking-wider font-extrabold">Activity Type</p>
                <p className="text-sm font-bold text-zinc-900 dark:text-white capitalize">{task.activity_type || 'Visit'}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-zinc-50 dark:bg-white/5 rounded-lg">
                <p className="text-xs text-zinc-900 dark:text-zinc-300 uppercase tracking-wider font-extrabold mb-1 flex items-center gap-1">
                  <Clock size={12} /> Time
                </p>
                <p className="text-sm font-bold text-zinc-900 dark:text-white">{task.start_time} - {task.end_time}</p>
              </div>
              <div className="p-3 bg-zinc-50 dark:bg-white/5 rounded-lg">
                <p className="text-xs text-zinc-900 dark:text-zinc-300 uppercase tracking-wider font-extrabold mb-1 flex items-center gap-1">
                  <MapPin size={12} /> Distance
                </p>
                <p className="text-sm font-bold text-zinc-900 dark:text-white">{task.distance_km ? `${task.distance_km} km` : 'N/A'}</p>
              </div>
            </div>

            <div className="p-3 bg-zinc-50 dark:bg-white/5 rounded-lg">
              <p className="text-xs text-zinc-900 dark:text-zinc-300 uppercase tracking-wider font-extrabold mb-1">Location</p>
              <p className="text-sm font-bold text-zinc-900 dark:text-white">{task.locality}</p>
            </div>
          </div>
        </div>
        <div className="p-4 border-t border-zinc-100 dark:border-white/5 bg-zinc-50/50 dark:bg-black/20 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-lg text-sm font-medium hover:opacity-90 transition"
          >
            Close
          </button>
        </div>
      </motion.div>
    </div>
  );
}

function TaskCard({ task, status, onUpdate, onClick, isEditable = true }) {
  const [processing, setProcessing] = useState(null);

  const handleUpdate = async (newStatus) => {
    setProcessing(newStatus);
    try {
      if (onUpdate) {
        await onUpdate(task.activity_id, newStatus);
      }
    } finally {
      // Small delay to ensure animation is visible if operation is instant, 
      // though mostly valuable if component doesn't unmount immediately.
      // If component unmounts (moved to another list), this update is ignored.
      setProcessing(null);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      onClick={() => status === 'Planned' && onClick && onClick(task)}
      className={`bg-white/60 dark:bg-zinc-900/60 backdrop-blur-md p-4 rounded-xl border border-zinc-200/60 dark:border-white/10 shadow-sm transition-all group ${status === 'Planned' ? 'cursor-pointer hover:shadow-md hover:border-zinc-300 dark:hover:border-white/20' : ''}`}
    >
      <h4 className="font-bold text-zinc-900 dark:text-zinc-100 mb-1 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
        {task.customer_name}
      </h4>
      <div className="space-y-1">
        <div className="flex items-center gap-2 text-xs font-bold text-zinc-700 dark:text-zinc-300">
          <Clock size={14} /> {task.start_time} - {task.end_time}
        </div>
        <div className="flex items-center gap-2 text-xs font-bold text-zinc-700 dark:text-zinc-300">
          <MapPin size={14} /> {task.locality}
        </div>
        {(task.travel_duration_min || task.distance_km) && (
          <div className="flex items-center gap-2 text-xs font-bold text-zinc-500 dark:text-zinc-400 mt-1">
            <span>🚗</span>
            <span>
              {task.distance_km ? `${task.distance_km} km` : ''}
              {task.distance_km && task.travel_duration_min ? ' • ' : ''}
              {task.travel_duration_min ? `${task.travel_duration_min} min` : ''}
            </span>
          </div>
        )}
      </div>

      {isEditable && (
        <div className="mt-4 flex gap-2 pt-3 border-t border-zinc-100 dark:border-white/5" onClick={e => e.stopPropagation()}>
          {status !== 'Completed' && (
            <button
              onClick={() => handleUpdate('Done')}
              disabled={!!processing}
              className={`flex-1 py-1.5 bg-gradient-to-r from-zinc-900 to-zinc-700 dark:from-zinc-100 dark:to-zinc-300 text-white dark:text-zinc-900 text-xs rounded font-bold hover:shadow-lg hover:scale-[1.02] transition-all shadow-sm flex items-center justify-center gap-2 ${processing === 'Done' ? 'opacity-80 cursor-wait' : ''}`}
            >
              {processing === 'Done' ? (
                <>
                  <Loader2 className="animate-spin" size={12} />
                  Processing...
                </>
              ) : (
                'Complete'
              )}
            </button>
          )}
          {status === 'Completed' && (
            <button
              onClick={() => handleUpdate('Pending')}
              disabled={!!processing}
              className={`flex-1 py-1.5 bg-gradient-to-r from-zinc-100 to-zinc-200 dark:from-zinc-800 dark:to-zinc-700 text-zinc-900 dark:text-zinc-300 text-xs rounded font-bold hover:bg-zinc-200 dark:hover:bg-zinc-700 transition flex items-center justify-center gap-2 ${processing === 'Pending' ? 'opacity-80 cursor-wait' : ''}`}
            >
              {processing === 'Pending' ? (
                <>
                  <Loader2 className="animate-spin" size={12} />
                  Reverting...
                </>
              ) : (
                'Revert'
              )}
            </button>
          )}
          {status !== 'Cancelled' && (
            <button
              onClick={() => handleUpdate('Cancelled')}
              disabled={!!processing}
              className={`px-3 py-1.5 bg-gradient-to-r from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-900/40 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-xs rounded font-bold hover:from-red-100 hover:to-red-200 transition flex items-center justify-center gap-2 ${processing === 'Cancelled' ? 'opacity-80 cursor-wait' : ''}`}
            >
              {processing === 'Cancelled' ? (
                <Loader2 className="animate-spin" size={12} />
              ) : (
                'Cancel'
              )}
            </button>
          )}
        </div>
      )}
    </motion.div>
  );
}