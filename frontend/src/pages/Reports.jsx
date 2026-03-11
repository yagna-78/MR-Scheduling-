import React, { useState, useEffect } from 'react';
import api from '../lib/api';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { BarChart, UserCheck, Globe, MapPin, Download, FileText, Calendar, AlertCircle } from 'lucide-react';

export default function Reports() {
  const { user, logout } = useAuth();
  const isAdmin = user?.role === 'admin';

  const [activeTab, setActiveTab] = useState('activity');
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [year, setYear] = useState(new Date().getFullYear());
  const [mrId, setMrId] = useState(isAdmin ? '' : user.user_id);
  const [mrs, setMrs] = useState([]);
  const [data, setData] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const tabs = [
    { id: 'activity', label: 'Activity Report', icon: BarChart, hasDateRange: true },
    { id: 'compliance', label: 'Compliance Report', icon: UserCheck, hasDateRange: false },
    { id: 'customer-behaviour', label: 'Customer Behaviour Report', icon: Globe, hasDateRange: false },
    { id: 'travel', label: 'Travel KM Report', icon: MapPin, hasDateRange: false },
  ];

  useEffect(() => {
    if (isAdmin) {
      api.get('/admin/mrs')
        .then(res => {
          setMrs(res.data || []);
          if (res.data && res.data.length > 0) {
            setMrId(res.data[0].mr_id);
          }
        })
        .catch(err => {
          console.error('Failed to load MRs:', err);
          setMrs([]);
        });
    }
  }, [isAdmin]);

  const fetchReport = async () => {
    setLoading(true);
    setError(null);
    setData([]);
    setTotalCount(0);

    try {
      let endpoint = `/reports/${activeTab}?mr_id=${mrId}`;
      if (tabs.find(t => t.id === activeTab).hasDateRange) {
        endpoint += `&start_date=${startDate}&end_date=${endDate}`;
      } else {
        endpoint += `&month=${month}&year=${year}`;
      }

      const response = await api.get(endpoint);
      setData(response.data.data || response.data || []);
      setTotalCount(response.data.total_count || response.data.length || 0);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load report');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setCurrentPage(1); // Reset to first page when filtering changes
    fetchReport();
  }, [activeTab, startDate, endDate, month, year, mrId]);

  const exportToCSV = () => {
    if (data.length === 0) return;

    // Remove unwanted keys like 'actions' or 'sr_no' for export
    const headers = Object.keys(data[0]).filter(key =>
      !['actions', 'sr_no'].includes(key.toLowerCase())
    );

    const csvContent = [
      headers.join(','),
      ...data.map(row =>
        headers.map(fieldName => {
          const value = row[fieldName] === null || row[fieldName] === undefined ? '' : row[fieldName];
          // Escape quotes and wrap in quotes if contains comma
          const stringValue = String(value).replace(/"/g, '""');
          return `"${stringValue}"`;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${activeTab}-report.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const columns = data.length > 0
    ? Object.keys(data[0]).filter(key =>
      !['actions', 'sr_no', 'compliance_dates'].includes(key.toLowerCase())
    )
    : [];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gradient-to-br dark:from-zinc-900 dark:to-zinc-950 p-6">
      <div className="w-full space-y-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">
              Reports
            </h1>
            <p className="text-zinc-900 dark:text-zinc-300 mt-1 font-medium">
              Analyze performance and compliance
            </p>
          </div>
          <button
            onClick={logout}
            className="px-6 py-2.5 bg-gradient-to-r from-red-600 to-red-500 text-white rounded-lg hover:from-red-500 hover:to-red-400 transition-all shadow-md text-sm font-bold flex items-center gap-2"
          >
            Sign Out
          </button>
        </div>

        {/* Tabs */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-5 py-3 rounded-xl font-bold text-sm transition-all border flex items-center justify-center gap-2 ${activeTab === tab.id
                ? 'bg-gradient-to-r from-zinc-900 to-zinc-700 dark:from-zinc-100 dark:to-zinc-300 text-white dark:text-zinc-900 border-zinc-900 dark:border-zinc-100 shadow-lg scale-[1.02]'
                : 'bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800 font-bold'
                }`}
            >
              <tab.icon size={18} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Filters */}
        <div className="flex flex-col md:flex-row gap-4 bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl p-5 rounded-xl border border-zinc-200/50 dark:border-white/10 shadow-sm">
          {tabs.find(t => t.id === activeTab).hasDateRange ? (
            <>
              <div className="flex items-center gap-3 flex-1">
                <label className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">Start:</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={e => setStartDate(e.target.value)}
                  className="px-3 py-2 border border-zinc-200 dark:border-zinc-700 rounded-lg bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-zinc-500 outline-none flex-1 text-sm font-semibold"
                />
              </div>
              <div className="flex items-center gap-3 flex-1">
                <label className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">End:</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={e => setEndDate(e.target.value)}
                  className="px-3 py-2 border border-zinc-200 dark:border-zinc-700 rounded-lg bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-zinc-500 outline-none flex-1 text-sm font-semibold"
                />
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-3 flex-1">
                <label className="font-extrabold text-zinc-900 dark:text-zinc-100 text-sm">Month:</label>
                <select
                  value={month}
                  onChange={e => setMonth(parseInt(e.target.value))}
                  className="px-3 py-2 border border-zinc-200 dark:border-zinc-700 rounded-lg bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-zinc-500 outline-none flex-1 text-sm font-bold"
                >
                  {Array.from({ length: 12 }, (_, i) => i + 1).map(m => (
                    <option key={m} value={m}>{new Date(0, m - 1).toLocaleString('default', { month: 'long' })}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-3 flex-1">
                <label className="font-extrabold text-zinc-900 dark:text-zinc-100 text-sm">Year:</label>
                <input
                  type="number"
                  value={year}
                  onChange={e => setYear(parseInt(e.target.value))}
                  className="px-3 py-2 border border-zinc-200 dark:border-zinc-700 rounded-lg bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-zinc-500 outline-none flex-1 text-sm font-bold"
                  min={2020}
                  max={2030}
                />
              </div>
            </>
          )}

          {isAdmin && (
            <div className="flex items-center gap-3 flex-1">
              <label className="font-bold text-zinc-900 dark:text-zinc-100 text-sm">MR:</label>
              <select
                value={mrId}
                onChange={e => setMrId(e.target.value)}
                className="px-3 py-2 border border-zinc-200 dark:border-zinc-700 rounded-lg bg-zinc-50 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-zinc-500 outline-none flex-1 text-sm font-semibold"
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

        {/* Report Content */}
        <div className="bg-white/50 dark:bg-zinc-900/50 backdrop-blur-xl p-8 rounded-xl border border-zinc-200/50 dark:border-white/10 shadow-sm min-h-[400px]">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-zinc-800 dark:border-zinc-200 border-t-transparent"></div>
              <p className="mt-4 text-sm text-zinc-700 dark:text-zinc-300">Generatng report...</p>
            </div>
          ) : error ? (
            <div className="text-center py-20 text-red-500">
              <AlertCircle size={48} className="mx-auto mb-4 opacity-80" />
              <p className="text-lg font-medium">{error}</p>
            </div>
          ) : data.length === 0 ? (
            <div className="text-center py-20 text-zinc-600 dark:text-zinc-400">
              <FileText size={64} className="mx-auto mb-4 opacity-50" />
              <p className="text-xl font-medium">No data available</p>
              <p className="mt-2 text-sm">Try adjusting your filters</p>
            </div>
          ) : (
            <>
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                <h2 className="text-xl font-bold text-zinc-800 dark:text-zinc-100 flex items-center gap-2">
                  {tabs.find(t => t.id === activeTab).label}
                </h2>
                <button
                  onClick={exportToCSV}
                  className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-indigo-500 text-white rounded-lg hover:from-indigo-500 hover:to-indigo-400 transition-all flex items-center gap-2 text-sm font-bold shadow-md"
                >
                  <Download size={16} />
                  Export CSV
                </button>
              </div>

              <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
                <table className="min-w-full divide-y divide-zinc-200 dark:divide-zinc-800">
                  <thead className="bg-zinc-900 text-white dark:bg-zinc-950">
                    <tr>
                      {columns.map(col => (
                        <th key={col} className="px-6 py-4 text-left text-xs font-black uppercase tracking-wider border-b border-zinc-700">
                          {col.replace('_', ' ').toUpperCase()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-zinc-900 divide-y divide-zinc-200 dark:divide-zinc-800">
                    {data.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage).map((row, index) => (
                      <tr key={index} className="hover:bg-blue-50 dark:hover:bg-blue-900/10 even:bg-zinc-50 dark:even:bg-white/5 transition-colors">
                        {columns.map(col => (
                          <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-zinc-900 dark:text-zinc-100 font-extrabold">
                            {row[col] ?? '-'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination Controls */}
              {data.length > itemsPerPage && (
                <div className="mt-6 flex flex-col sm:flex-row justify-between items-center gap-4 text-sm text-zinc-700 dark:text-zinc-300">
                  <p>
                    Showing {(currentPage - 1) * itemsPerPage + 1}–{Math.min(currentPage * itemsPerPage, data.length)} of {data.length}
                  </p>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                      disabled={currentPage === 1}
                      className={`px-3 py-1.5 rounded border border-zinc-200 dark:border-zinc-700 flex items-center gap-1 transition-colors ${currentPage === 1
                        ? 'bg-zinc-50 dark:bg-zinc-800 text-zinc-300 dark:text-zinc-600 cursor-not-allowed'
                        : 'bg-white dark:bg-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300'
                        }`}
                    >
                      Previous
                    </button>

                    <div className="flex items-center gap-1">
                      {Array.from({ length: Math.min(5, Math.ceil(data.length / itemsPerPage)) }, (_, i) => {
                        let pageNum;
                        const totalPages = Math.ceil(data.length / itemsPerPage);
                        if (totalPages <= 5) {
                          pageNum = i + 1;
                        } else if (currentPage <= 3) {
                          pageNum = i + 1;
                        } else if (currentPage >= totalPages - 2) {
                          pageNum = totalPages - 4 + i;
                        } else {
                          pageNum = currentPage - 2 + i;
                        }

                        return (
                          <button
                            key={pageNum}
                            onClick={() => setCurrentPage(pageNum)}
                            className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs transition-all ${currentPage === pageNum
                              ? 'bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900'
                              : 'hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-600 dark:text-zinc-400'
                              }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>

                    <button
                      onClick={() => setCurrentPage(prev => Math.min(prev + 1, Math.ceil(data.length / itemsPerPage)))}
                      disabled={currentPage === Math.ceil(data.length / itemsPerPage)}
                      className={`px-3 py-1.5 rounded border border-zinc-200 dark:border-zinc-700 flex items-center gap-1 transition-colors ${currentPage === Math.ceil(data.length / itemsPerPage)
                        ? 'bg-zinc-50 dark:bg-zinc-800 text-zinc-300 dark:text-zinc-600 cursor-not-allowed'
                        : 'bg-white dark:bg-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300'
                        }`}
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}

              {activeTab === 'activity' && totalCount > 0 && (
                <div className="mt-6 flex justify-end gap-8 border-t border-zinc-100 dark:border-zinc-800 pt-4">
                  <p className="text-lg font-medium text-zinc-700 dark:text-zinc-300">
                    Total: <span className="text-zinc-900 dark:text-white font-bold">{totalCount}</span>
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}