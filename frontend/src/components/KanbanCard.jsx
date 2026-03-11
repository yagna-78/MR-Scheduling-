import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { motion } from 'framer-motion';
import { Clock, MapPin, Phone, User, MessageSquare } from 'lucide-react';

export default function KanbanCard({ task, column }) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
  } = useSortable({ id: task.activity_id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`p-4 rounded-lg border cursor-grab active:cursor-grabbing shadow-sm ${column === 'completed'
        ? 'bg-gray-50 dark:bg-gray-700/50 line-through opacity-80'
        : 'bg-white dark:bg-gray-800 border-l-4 border-l-indigo-500'
        }`}
    >
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-medium">{task.customer_name}</h3>
        <span className="text-xs px-2 py-1 rounded-full bg-indigo-100 dark:bg-indigo-900 text-indigo-800 dark:text-indigo-200">
          {task.activity_type}
        </span>
      </div>

      {task.contact_person && (
        <div className="flex items-center text-sm text-gray-700 dark:text-gray-200 mb-1 font-medium">
          <User size={14} className="mr-1.5" />
          Dr. {task.contact_person}
        </div>
      )}

      <div className="flex items-center text-sm text-gray-700 dark:text-gray-200 mb-1 font-medium">
        <Clock size={14} className="mr-1.5" />
        {task.start_time} – {task.end_time}
      </div>

      <div className="flex items-center text-sm text-gray-700 dark:text-gray-200 mb-2 font-medium">
        <MapPin size={14} className="mr-1.5" />
        {task.locality}
      </div>

      {task.phone && task.phone !== 'N/A' && (
        <div className="flex items-center text-sm text-gray-700 dark:text-gray-200 mb-2 font-medium">
          <Phone size={14} className="mr-1.5" />
          <a href={`tel:${task.phone}`} className="hover:underline">{task.phone}</a>
        </div>
      )}

      {task.suggested_talking_points && (
        <div className="mt-3 pt-3 border-t dark:border-gray-700">
          <div className="flex items-start text-sm">
            <MessageSquare size={14} className="mr-1.5 mt-0.5 text-indigo-600 dark:text-indigo-400" />
            <p className="text-gray-800 dark:text-gray-200 font-medium">{task.suggested_talking_points}</p>
          </div>
        </div>
      )}

      {(task.distance_km || task.travel_duration_min) && (
        <div className="mt-3 text-xs text-gray-700 dark:text-gray-300 font-medium flex items-center gap-1">
          <span>🚗</span>
          {task.distance_km ? `≈ ${task.distance_km} km` : ''}
          {task.distance_km && task.travel_duration_min ? ' • ' : ''}
          {task.travel_duration_min ? `${task.travel_duration_min} min travel` : ''}
        </div>
      )}
    </motion.div>
  );
}