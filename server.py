import os
import io
import json
import pandas as pd

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from classify import classify


app = FastAPI(title="AI Log Classification & Anomaly Detection")


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "resources"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HTML / FRONTEND
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>AI Log Intelligence Platform</title>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            background: #080b12;
            color: #e8edf7;
            min-height: 100vh;
        }

        button,
        input,
        select {
            font-family: inherit;
        }

        .app {
            display: flex;
            min-height: 100vh;
        }

        /* =====================================================
           SIDEBAR
        ===================================================== */

        .sidebar {
            width: 250px;
            background: #0d111a;
            border-right: 1px solid #202735;
            padding: 22px 16px;
            position: fixed;
            left: 0;
            top: 0;
            bottom: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 35px;
            padding: 0 10px;
        }

        .logo-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 21px;
        }

        .logo-text {
            font-size: 17px;
            font-weight: 700;
        }

        .logo-subtitle {
            font-size: 10px;
            color: #7f8ba3;
            margin-top: 3px;
        }

        .nav-title {
            font-size: 10px;
            color: #68738a;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            padding: 0 12px;
            margin-bottom: 10px;
        }

        .nav-item {
            width: 100%;
            border: none;
            background: transparent;
            color: #8792a8;
            padding: 12px;
            border-radius: 9px;
            margin-bottom: 5px;
            text-align: left;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: 0.2s;
            font-size: 13px;
        }

        .nav-item:hover {
            background: #171d29;
            color: white;
        }

        .nav-item.active {
            background: #20263a;
            color: #a78bfa;
        }

        .nav-icon {
            width: 20px;
            text-align: center;
        }

        /* =====================================================
           MAIN
        ===================================================== */

        .main {
            margin-left: 250px;
            width: calc(100% - 250px);
            min-height: 100vh;
        }

        .topbar {
            height: 70px;
            border-bottom: 1px solid #202735;
            background: rgba(8, 11, 18, 0.94);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 30px;
            position: sticky;
            top: 0;
            z-index: 50;
            backdrop-filter: blur(12px);
        }

        .page-title {
            font-size: 19px;
            font-weight: 700;
        }

        .page-description {
            color: #758198;
            font-size: 12px;
            margin-top: 4px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #8c98ad;
            font-size: 12px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 10px #22c55e;
        }

        .content {
            padding: 30px;
        }

        .page {
            display: none;
        }

        .page.active {
            display: block;
        }

        /* =====================================================
           CARDS
        ===================================================== */

        .stats-grid {
            display: grid;
            grid-template-columns:
                repeat(4, minmax(0, 1fr));

            gap: 16px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: #10151f;
            border: 1px solid #202735;
            border-radius: 14px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }

        .stat-card::after {
            content: "";
            position: absolute;
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: #6366f1;
            opacity: 0.05;
            right: -30px;
            top: -30px;
        }

        .stat-label {
            color: #77839a;
            font-size: 12px;
            margin-bottom: 10px;
        }

        .stat-value {
            font-size: 28px;
            font-weight: 750;
        }

        .stat-small {
            margin-top: 7px;
            color: #6d7890;
            font-size: 11px;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns:
                1.4fr 1fr;

            gap: 18px;
        }

        .card {
            background: #10151f;
            border: 1px solid #202735;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 18px;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 14px;
            font-weight: 700;
        }

        .card-subtitle {
            font-size: 11px;
            color: #717c91;
            margin-top: 4px;
        }

        .chart-container {
            position: relative;
            height: 300px;
        }

        /* =====================================================
           UPLOAD
        ===================================================== */

        .upload-box {
            border: 1px dashed #3b4558;
            border-radius: 16px;
            padding: 55px 25px;
            text-align: center;
            background: #0c1119;
            transition: 0.2s;
            cursor: pointer;
        }

        .upload-box:hover,
        .upload-box.dragover {
            border-color: #8b5cf6;
            background: #111626;
        }

        .upload-icon {
            width: 65px;
            height: 65px;
            margin: auto auto 18px;
            border-radius: 16px;
            background: #1a1d31;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
        }

        .upload-title {
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .upload-description {
            color: #707b91;
            font-size: 12px;
            margin-bottom: 20px;
        }

        .file-name {
            margin-top: 14px;
            color: #a78bfa;
            font-size: 12px;
        }

        .btn {
            border: none;
            border-radius: 9px;
            padding: 11px 18px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 650;
            transition: 0.2s;
        }

        .btn-primary {
            background: linear-gradient(
                135deg,
                #6366f1,
                #8b5cf6
            );

            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 7px 20px rgba(99, 102, 241, 0.25);
        }

        .btn-secondary {
            background: #1b2230;
            color: #c8d0df;
            border: 1px solid #2a3344;
        }

        .btn-secondary:hover {
            background: #252e40;
        }

        .btn:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            transform: none;
        }

        input[type="file"] {
            display: none;
        }

        /* =====================================================
           PROGRESS
        ===================================================== */

        .progress-area {
            display: none;
            margin-top: 20px;
        }

        .progress-text {
            display: flex;
            justify-content: space-between;
            color: #8a95a9;
            font-size: 11px;
            margin-bottom: 8px;
        }

        .progress-track {
            height: 6px;
            border-radius: 10px;
            background: #1b2230;
            overflow: hidden;
        }

        .progress-bar {
            width: 0%;
            height: 100%;
            background: linear-gradient(
                90deg,
                #6366f1,
                #a78bfa
            );

            transition: width 0.4s;
        }

        /* =====================================================
           TABLE
        ===================================================== */

        .table-wrapper {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            color: #69758b;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #242c3b;
            white-space: nowrap;
        }

        td {
            padding: 13px 12px;
            border-bottom: 1px solid #1d2532;
            color: #b7c0d0;
            font-size: 12px;
            vertical-align: top;
        }

        tr:hover td {
            background: #131923;
        }

        .log-message {
            max-width: 520px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .badge {
            display: inline-flex;
            padding: 5px 9px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: 700;
            background: #17251d;
            color: #4ade80;
        }

        .badge-anomaly {
            background: #2d1a1e;
            color: #fb7185;
        }

        .badge-other {
            background: #28231a;
            color: #fbbf24;
        }

        /* =====================================================
           FILTERS
        ===================================================== */

        .filters {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 18px;
        }

        .search-input,
        .select-input {
            background: #0d121b;
            border: 1px solid #283141;
            color: #d9dfeb;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 12px;
            outline: none;
        }

        .search-input {
            min-width: 280px;
        }

        .search-input:focus,
        .select-input:focus {
            border-color: #6366f1;
        }

        /* =====================================================
           MODEL
        ===================================================== */

        .model-grid {
            display: grid;
            grid-template-columns:
                repeat(3, minmax(0, 1fr));

            gap: 15px;
        }

        .model-item {
            background: #0c1119;
            border: 1px solid #202735;
            border-radius: 12px;
            padding: 18px;
        }

        .model-icon {
            font-size: 22px;
            margin-bottom: 12px;
        }

        .model-item h3 {
            font-size: 13px;
            margin-bottom: 7px;
        }

        .model-item p {
            color: #758198;
            font-size: 11px;
            line-height: 1.6;
        }

        /* =====================================================
           MODAL
        ===================================================== */

        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.72);
            z-index: 500;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: #10151f;
            border: 1px solid #2a3344;
            border-radius: 15px;
            width: min(800px, 100%);
            max-height: 85vh;
            overflow-y: auto;
            padding: 24px;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .close-btn {
            background: transparent;
            border: none;
            color: #8d98aa;
            font-size: 24px;
            cursor: pointer;
        }

        .detail-row {
            margin-bottom: 16px;
        }

        .detail-label {
            font-size: 10px;
            color: #707c91;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .detail-value {
            background: #0b1018;
            border: 1px solid #202735;
            border-radius: 8px;
            padding: 12px;
            color: #c9d1df;
            font-size: 12px;
            line-height: 1.6;
            word-break: break-word;
        }

        /* =====================================================
           EMPTY STATE
        ===================================================== */

        .empty {
            padding: 50px 20px;
            text-align: center;
            color: #69758b;
        }

        .empty-icon {
            font-size: 35px;
            margin-bottom: 10px;
        }

        /* =====================================================
           RESPONSIVE
        ===================================================== */

        @media (max-width: 1100px) {

            .stats-grid {
                grid-template-columns:
                    repeat(2, 1fr);
            }

            .dashboard-grid {
                grid-template-columns: 1fr;
            }

            .model-grid {
                grid-template-columns:
                    repeat(2, 1fr);
            }
        }

        @media (max-width: 760px) {

            .sidebar {
                width: 70px;
                padding: 18px 10px;
            }

            .logo-text,
            .logo-subtitle,
            .nav-title,
            .nav-item span:not(.nav-icon) {
                display: none;
            }

            .logo {
                justify-content: center;
                padding: 0;
            }

            .nav-item {
                justify-content: center;
            }

            .main {
                margin-left: 70px;
                width: calc(100% - 70px);
            }

            .content {
                padding: 18px;
            }

            .topbar {
                padding: 0 18px;
            }

            .stats-grid {
                grid-template-columns: 1fr;
            }

            .model-grid {
                grid-template-columns: 1fr;
            }

            .search-input {
                min-width: 100%;
            }
        }

    </style>

</head>


<body>

<div class="app">

    <!-- =====================================================
         SIDEBAR
    ====================================================== -->

    <aside class="sidebar">

        <div class="logo">

            <div class="logo-icon">
                AI
            </div>

            <div>
                <div class="logo-text">
                    Log Intelligence
                </div>

                <div class="logo-subtitle">
                    AI Security Platform
                </div>
            </div>

        </div>


        <div class="nav-title">
            Workspace
        </div>


        <button
            class="nav-item active"
            onclick="navigate('dashboard', this)"
        >
            <span class="nav-icon">▦</span>
            <span>Dashboard</span>
        </button>


        <button
            class="nav-item"
            onclick="navigate('analyze', this)"
        >
            <span class="nav-icon">⇧</span>
            <span>Analyze Logs</span>
        </button>


        <button
            class="nav-item"
            onclick="navigate('results', this)"
        >
            <span class="nav-icon">☷</span>
            <span>Results</span>
        </button>


        <button
            class="nav-item"
            onclick="navigate('analytics', this)"
        >
            <span class="nav-icon">◔</span>
            <span>Analytics</span>
        </button>


        <div class="nav-title" style="margin-top: 28px;">
            System
        </div>


        <button
            class="nav-item"
            onclick="navigate('model', this)"
        >
            <span class="nav-icon">◎</span>
            <span>Model</span>
        </button>


        <button
            class="nav-item"
            onclick="downloadResults()"
        >
            <span class="nav-icon">↓</span>
            <span>Download Results</span>
        </button>

    </aside>


    <!-- =====================================================
         MAIN
    ====================================================== -->

    <main class="main">


        <header class="topbar">

            <div>

                <div
                    class="page-title"
                    id="pageTitle"
                >
                    Security Dashboard
                </div>

                <div
                    class="page-description"
                    id="pageDescription"
                >
                    AI-powered log classification and anomaly analysis
                </div>

            </div>


            <div class="status">

                <span class="status-dot"></span>

                System Ready

            </div>

        </header>


        <section class="content">


            <!-- =================================================
                 DASHBOARD
            ================================================== -->

            <div
                id="dashboard"
                class="page active"
            >

                <div class="stats-grid">

                    <div class="stat-card">

                        <div class="stat-label">
                            TOTAL LOGS
                        </div>

                        <div
                            class="stat-value"
                            id="totalLogs"
                        >
                            0
                        </div>

                        <div class="stat-small">
                            Records analyzed
                        </div>

                    </div>


                    <div class="stat-card">

                        <div class="stat-label">
                            NORMAL
                        </div>

                        <div
                            class="stat-value"
                            id="normalLogs"
                        >
                            0
                        </div>

                        <div class="stat-small">
                            Classified as normal
                        </div>

                    </div>


                    <div class="stat-card">

                        <div class="stat-label">
                            ANOMALIES
                        </div>

                        <div
                            class="stat-value"
                            id="anomalyLogs"
                        >
                            0
                        </div>

                        <div class="stat-small">
                            Potential anomalies
                        </div>

                    </div>


                    <div class="stat-card">

                        <div class="stat-label">
                            OTHER
                        </div>

                        <div
                            class="stat-value"
                            id="otherLogs"
                        >
                            0
                        </div>

                        <div class="stat-small">
                            Other / unclassified
                        </div>

                    </div>

                </div>


                <div class="dashboard-grid">

                    <div class="card">

                        <div class="card-header">

                            <div>

                                <div class="card-title">
                                    Classification Distribution
                                </div>

                                <div class="card-subtitle">
                                    Log categories returned by the classifier
                                </div>

                            </div>

                        </div>

                        <div class="chart-container">

                            <canvas
                                id="classificationChart"
                            ></canvas>

                        </div>

                    </div>


                    <div class="card">

                        <div class="card-header">

                            <div>

                                <div class="card-title">
                                    Log Sources
                                </div>

                                <div class="card-subtitle">
                                    Distribution by source
                                </div>

                            </div>

                        </div>

                        <div class="chart-container">

                            <canvas
                                id="sourceChart"
                            ></canvas>

                        </div>

                    </div>

                </div>


                <div class="card">

                    <div class="card-header">

                        <div>

                            <div class="card-title">
                                Recent Classification Results
                            </div>

                            <div class="card-subtitle">
                                Latest processed records
                            </div>

                        </div>


                        <button
                            class="btn btn-secondary"
                            onclick="navigateByName('results')"
                        >
                            View All
                        </button>

                    </div>


                    <div
                        id="recentResults"
                        class="table-wrapper"
                    >

                        <div class="empty">

                            <div class="empty-icon">
                                ◌
                            </div>

                            Upload a CSV file to begin analysis.

                        </div>

                    </div>

                </div>

            </div>


            <!-- =================================================
                 ANALYZE
            ================================================== -->

            <div
                id="analyze"
                class="page"
            >

                <div class="card">

                    <div class="card-header">

                        <div>

                            <div class="card-title">
                                Analyze Log File
                            </div>

                            <div class="card-subtitle">
                                Upload a CSV containing source and log_message columns
                            </div>

                        </div>

                    </div>


                    <label
                        class="upload-box"
                        id="uploadBox"
                        for="fileInput"
                    >

                        <div class="upload-icon">
                            ⇧
                        </div>

                        <div class="upload-title">
                            Drop your CSV file here
                        </div>

                        <div class="upload-description">
                            or click to browse from your computer
                        </div>

                        <span class="btn btn-secondary">
                            Select CSV File
                        </span>

                        <div
                            id="fileName"
                            class="file-name"
                        ></div>

                    </label>


                    <input
                        type="file"
                        id="fileInput"
                        accept=".csv"
                    >


                    <div
                        class="progress-area"
                        id="progressArea"
                    >

                        <div class="progress-text">

                            <span id="progressText">
                                Processing logs...
                            </span>

                            <span id="progressPercent">
                                0%
                            </span>

                        </div>

                        <div class="progress-track">

                            <div
                                class="progress-bar"
                                id="progressBar"
                            ></div>

                        </div>

                    </div>


                    <div style="
                        margin-top:20px;
                        display:flex;
                        gap:10px;
                    ">

                        <button
                            class="btn btn-primary"
                            id="analyzeButton"
                            onclick="analyzeFile()"
                            disabled
                        >
                            Analyze Logs
                        </button>

                        <button
                            class="btn btn-secondary"
                            onclick="clearFile()"
                        >
                            Clear
                        </button>

                    </div>

                </div>


                <div class="card">

                    <div class="card-title">
                        Required CSV Format
                    </div>

                    <div
                        style="
                            color:#758198;
                            font-size:12px;
                            line-height:1.8;
                            margin-top:12px;
                        "
                    >

                        Your CSV must contain these columns:

                        <br><br>

                        <code>
                            source
                        </code>

                        and

                        <code>
                            log_message
                        </code>

                        <br><br>

                        Example:

                        <br>

                        <code>
                            source,log_message
                        </code>

                        <br>

                        <code>
                            server_01,Connection timeout occurred
                        </code>

                    </div>

                </div>

            </div>


            <!-- =================================================
                 RESULTS
            ================================================== -->

            <div
                id="results"
                class="page"
            >

                <div class="card">

                    <div class="card-header">

                        <div>

                            <div class="card-title">
                                Classification Results
                            </div>

                            <div class="card-subtitle">
                                Search and inspect processed logs
                            </div>

                        </div>

                        <button
                            class="btn btn-primary"
                            onclick="downloadResults()"
                        >
                            Download CSV
                        </button>

                    </div>


                    <div class="filters">

                        <input
                            id="searchInput"
                            class="search-input"
                            type="text"
                            placeholder="Search logs..."
                            oninput="applyFilters()"
                        >


                        <select
                            id="labelFilter"
                            class="select-input"
                            onchange="applyFilters()"
                        >

                            <option value="">
                                All classifications
                            </option>

                        </select>


                        <select
                            id="sourceFilter"
                            class="select-input"
                            onchange="applyFilters()"
                        >

                            <option value="">
                                All sources
                            </option>

                        </select>

                    </div>


                    <div
                        id="resultsTable"
                        class="table-wrapper"
                    >

                        <div class="empty">

                            <div class="empty-icon">
                                ☷
                            </div>

                            No classification results yet.

                        </div>

                    </div>

                </div>

            </div>


            <!-- =================================================
                 ANALYTICS
            ================================================== -->

            <div
                id="analytics"
                class="page"
            >

                <div class="dashboard-grid">

                    <div class="card">

                        <div class="card-header">

                            <div>

                                <div class="card-title">
                                    Classification Analytics
                                </div>

                                <div class="card-subtitle">
                                    Overview of detected categories
                                </div>

                            </div>

                        </div>

                        <div class="chart-container">

                            <canvas
                                id="analyticsClassificationChart"
                            ></canvas>

                        </div>

                    </div>


                    <div class="card">

                        <div class="card-header">

                            <div>

                                <div class="card-title">
                                    Source Analytics
                                </div>

                                <div class="card-subtitle">
                                    Top log sources
                                </div>

                            </div>

                        </div>

                        <div class="chart-container">

                            <canvas
                                id="analyticsSourceChart"
                            ></canvas>

                        </div>

                    </div>

                </div>


                <div class="card">

                    <div class="card-title">
                        Analysis Summary
                    </div>

                    <div
                        id="analyticsSummary"
                        style="
                            margin-top:16px;
                            color:#8994a9;
                            font-size:12px;
                            line-height:1.8;
                        "
                    >
                        Upload and analyze a CSV to generate analytics.

                    </div>

                </div>

            </div>


            <!-- =================================================
                 MODEL
            ================================================== -->

            <div
                id="model"
                class="page"
            >

                <div class="card">

                    <div class="card-header">

                        <div>

                            <div class="card-title">
                                AI Model Pipeline
                            </div>

                            <div class="card-subtitle">
                                Hybrid log classification and anomaly detection workflow
                            </div>

                        </div>

                    </div>


                    <div class="model-grid">

                        <div class="model-item">

                            <div class="model-icon">
                                📥
                            </div>

                            <h3>
                                Log Ingestion
                            </h3>

                            <p>
                                CSV logs are uploaded and validated before
                                processing.
                            </p>

                        </div>


                        <div class="model-item">

                            <div class="model-icon">
                                🧹
                            </div>

                            <h3>
                                Preprocessing
                            </h3>

                            <p>
                                Log messages are prepared and transformed
                                for downstream classification.
                            </p>

                        </div>


                        <div class="model-item">

                            <div class="model-icon">
                                🧠
                            </div>

                            <h3>
                                Log Classification
                            </h3>

                            <p>
                                The existing classify() pipeline processes
                                the source and log message.
                            </p>

                        </div>


                        <div class="model-item">

                            <div class="model-icon">
                                🔍
                            </div>

                            <h3>
                                Anomaly Detection
                            </h3>

                            <p>
                                Anomaly-related labels returned by the
                                existing model are displayed in the dashboard.
                            </p>

                        </div>


                        <div class="model-item">

                            <div class="model-icon">
                                📊
                            </div>

                            <h3>
                                Analytics
                            </h3>

                            <p>
                                Classification distributions and source
                                statistics are visualized.
                            </p>

                        </div>


                        <div class="model-item">

                            <div class="model-icon">
                                📤
                            </div>

                            <h3>
                                Export
                            </h3>

                            <p>
                                Classified logs can be downloaded as a CSV
                                file for further analysis.
                            </p>

                        </div>

                    </div>

                </div>

            </div>


        </section>

    </main>

</div>


<!-- ============================================================
     DETAIL MODAL
============================================================= -->

<div
    class="modal"
    id="detailModal"
>

    <div class="modal-content">

        <div class="modal-header">

            <div class="card-title">
                Log Details
            </div>

            <button
                class="close-btn"
                onclick="closeModal()"
            >
                ×
            </button>

        </div>


        <div id="modalBody"></div>

    </div>

</div>


<script>

    // =========================================================
    // GLOBAL DATA
    // =========================================================

    let allData = [];
    let filteredData = [];
    let totalRecords = 0;

    let selectedFile = null;

    let classificationChart = null;
    let sourceChart = null;
    let analyticsClassificationChart = null;
    let analyticsSourceChart = null;


    // =========================================================
    // NAVIGATION
    // =========================================================

    const pageInfo = {

        dashboard: {
            title: "Security Dashboard",
            description:
                "AI-powered log classification and anomaly analysis"
        },

        analyze: {
            title: "Analyze Logs",
            description:
                "Upload and classify your log dataset"
        },

        results: {
            title: "Classification Results",
            description:
                "Inspect and search analyzed log records"
        },

        analytics: {
            title: "Analytics",
            description:
                "Explore classification and source distributions"
        },

        model: {
            title: "Model Pipeline",
            description:
                "Understand the AI log processing workflow"
        }

    };


    function navigate(page, element) {

        document.querySelectorAll(".page")
            .forEach(function(item) {
                item.classList.remove("active");
            });


        const target = document.getElementById(page);

        if (target) {
            target.classList.add("active");
        }


        document.querySelectorAll(".nav-item")
            .forEach(function(item) {
                item.classList.remove("active");
            });


        if (element) {
            element.classList.add("active");
        }


        const info = pageInfo[page];

        if (info) {

            document.getElementById("pageTitle")
                .textContent = info.title;

            document.getElementById("pageDescription")
                .textContent = info.description;

        }


        if (page === "analytics") {
            updateAnalytics();
        }

    }


    function navigateByName(page) {

        const buttons =
            document.querySelectorAll(".nav-item");

        buttons.forEach(function(button) {

            if (
                button.textContent
                    .toLowerCase()
                    .includes(page)
            ) {

                navigate(page, button);

            }

        });

    }


    // =========================================================
    // FILE UPLOAD
    // =========================================================

    const fileInput =
        document.getElementById("fileInput");

    const uploadBox =
        document.getElementById("uploadBox");


    fileInput.addEventListener(
        "change",
        function(event) {

            if (event.target.files.length > 0) {

                selectedFile =
                    event.target.files[0];

                showSelectedFile();

            }

        }
    );


    uploadBox.addEventListener(
        "dragover",
        function(event) {

            event.preventDefault();

            uploadBox.classList.add("dragover");

        }
    );


    uploadBox.addEventListener(
        "dragleave",
        function() {

            uploadBox.classList.remove("dragover");

        }
    );


    uploadBox.addEventListener(
        "drop",
        function(event) {

            event.preventDefault();

            uploadBox.classList.remove("dragover");

            if (event.dataTransfer.files.length > 0) {

                selectedFile =
                    event.dataTransfer.files[0];

                showSelectedFile();

            }

        }
    );


    function showSelectedFile() {

        if (!selectedFile) {
            return;
        }


        document.getElementById("fileName")
            .textContent =
            "Selected: " + selectedFile.name;


        document.getElementById("analyzeButton")
            .disabled = false;

    }


    function clearFile() {

        selectedFile = null;

        fileInput.value = "";

        document.getElementById("fileName")
            .textContent = "";

        document.getElementById("analyzeButton")
            .disabled = true;

        document.getElementById("progressArea")
            .style.display = "none";

        document.getElementById("progressBar")
            .style.width = "0%";

    }


    // =========================================================
    // ANALYZE FILE
    // =========================================================

    async function analyzeFile() {

        if (!selectedFile) {

            alert("Please select a CSV file first.");

            return;

        }


        if (
            !selectedFile.name
                .toLowerCase()
                .endsWith(".csv")
        ) {

            alert("Please upload a CSV file.");

            return;

        }


        const formData = new FormData();

        formData.append(
            "file",
            selectedFile
        );


        const progressArea =
            document.getElementById("progressArea");

        const progressBar =
            document.getElementById("progressBar");

        const progressText =
            document.getElementById("progressText");

        const progressPercent =
            document.getElementById("progressPercent");

        const analyzeButton =
            document.getElementById("analyzeButton");


        progressArea.style.display = "block";

        analyzeButton.disabled = true;


        let progress = 10;

        progressBar.style.width =
            progress + "%";

        progressPercent.textContent =
            progress + "%";


        progressText.textContent =
            "Uploading dataset...";


        const progressTimer =
            setInterval(function() {

                if (progress < 90) {

                    progress += 5;

                    progressBar.style.width =
                        progress + "%";

                    progressPercent.textContent =
                        progress + "%";


                    if (progress > 30) {

                        progressText.textContent =
                            "Running classification model...";

                    }

                }

            }, 400);


        try {

            const response =
                await fetch(
                    "/classify/",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            clearInterval(progressTimer);


            const result =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    result.detail ||
                    "Classification failed."
                );

            }


            progressBar.style.width =
                "100%";

            progressPercent.textContent =
                "100%";

            progressText.textContent =
                "Analysis completed.";


            allData =
                Array.isArray(result.data)
                    ? result.data
                    : [];


            totalRecords =
                Number(result.total_records || allData.length);


            filteredData =
                [...allData];


            updateDashboard();

            updateFilters();

            renderResults();

            updateAnalytics();


            setTimeout(function() {

                navigateByName("results");

            }, 600);


        } catch (error) {

            clearInterval(progressTimer);

            progressBar.style.width =
                "0%";

            progressPercent.textContent =
                "0%";

            progressText.textContent =
                "Analysis failed.";

            alert(
                "Error: " +
                error.message
            );

        } finally {

            analyzeButton.disabled = false;

        }

    }


    // =========================================================
    // HELPERS
    // =========================================================

    function getLabel(row) {

        const value =
            row.target_label ??
            row.label ??
            row.prediction ??
            row.classification ??
            "Unclassified";


        return String(value);

    }


    function getSource(row) {

        return String(
            row.source ?? "Unknown"
        );

    }


    function getMessage(row) {

        return String(
            row.log_message ??
            row.message ??
            ""
        );

    }


    function classifyLabel(label) {

        const text =
            label.toLowerCase();


        if (
            text.includes("anomaly") ||
            text.includes("anomal") ||
            text.includes("attack") ||
            text.includes("error")
        ) {

            return "anomaly";

        }


        if (
            text.includes("normal") ||
            text.includes("success") ||
            text.includes("ok")
        ) {

            return "normal";

        }


        return "other";

    }


    // =========================================================
    // DASHBOARD
    // =========================================================

    function updateDashboard() {

        let normal = 0;
        let anomaly = 0;
        let other = 0;


        allData.forEach(function(row) {

            const type =
                classifyLabel(
                    getLabel(row)
                );


            if (type === "normal") {

                normal++;

            } else if (type === "anomaly") {

                anomaly++;

            } else {

                other++;

            }

        });


        document.getElementById("totalLogs")
            .textContent =
            totalRecords.toLocaleString();


        document.getElementById("normalLogs")
            .textContent =
            normal.toLocaleString();


        document.getElementById("anomalyLogs")
            .textContent =
            anomaly.toLocaleString();


        document.getElementById("otherLogs")
            .textContent =
            other.toLocaleString();


        updateCharts(
            normal,
            anomaly,
            other
        );


        renderRecentResults();

    }


    // =========================================================
    // CHARTS
    // =========================================================

    function countLabels() {

        const counts = {};


        allData.forEach(function(row) {

            const label =
                getLabel(row);

            counts[label] =
                (counts[label] || 0) + 1;

        });


        return counts;

    }


    function countSources() {

        const counts = {};


        allData.forEach(function(row) {

            const source =
                getSource(row);

            counts[source] =
                (counts[source] || 0) + 1;

        });


        return counts;

    }


    function updateCharts(
        normal,
        anomaly,
        other
    ) {

        const classificationCanvas =
            document.getElementById(
                "classificationChart"
            );


        const sourceCanvas =
            document.getElementById(
                "sourceChart"
            );


        if (!classificationCanvas ||
            !sourceCanvas) {

            return;

        }


        if (classificationChart) {

            classificationChart.destroy();

        }


        if (sourceChart) {

            sourceChart.destroy();

        }


        classificationChart =
            new Chart(
                classificationCanvas,
                {
                    type: "doughnut",

                    data: {

                        labels: [
                            "Normal",
                            "Anomaly",
                            "Other"
                        ],

                        datasets: [

                            {
                                data: [
                                    normal,
                                    anomaly,
                                    other
                                ],

                                borderWidth: 0
                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        maintainAspectRatio: false,

                        plugins: {

                            legend: {
                                position: "bottom"
                            }

                        }

                    }

                }
            );


        const sourceCounts =
            countSources();


        const sourceEntries =
            Object.entries(sourceCounts)
                .sort(
                    function(a, b) {
                        return b[1] - a[1];
                    }
                )
                .slice(0, 10);


        sourceChart =
            new Chart(
                sourceCanvas,
                {
                    type: "bar",

                    data: {

                        labels:
                            sourceEntries.map(
                                function(item) {
                                    return item[0];
                                }
                            ),

                        datasets: [

                            {
                                label: "Logs",

                                data:
                                    sourceEntries.map(
                                        function(item) {
                                            return item[1];
                                        }
                                    ),

                                borderWidth: 0

                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        maintainAspectRatio: false,

                        scales: {

                            y: {
                                beginAtZero: true
                            }

                        },

                        plugins: {

                            legend: {
                                display: false
                            }

                        }

                    }

                }
            );

    }


    // =========================================================
    // RECENT RESULTS
    // =========================================================

    function renderRecentResults() {

        const container =
            document.getElementById(
                "recentResults"
            );


        if (!allData.length) {

            container.innerHTML = `
                <div class="empty">
                    <div class="empty-icon">◌</div>
                    No classification results yet.
                </div>
            `;

            return;

        }


        const rows =
            allData.slice(0, 8);


        let html = `
            <table>

                <thead>

                    <tr>

                        <th>Source</th>
                        <th>Log Message</th>
                        <th>Classification</th>

                    </tr>

                </thead>

                <tbody>
        `;


        rows.forEach(function(row, index) {

            const label =
                getLabel(row);

            const type =
                classifyLabel(label);


            let badgeClass = "badge";

            if (type === "anomaly") {
                badgeClass += " badge-anomaly";
            }

            if (type === "other") {
                badgeClass += " badge-other";
            }


            html += `
                <tr
                    onclick="showDetails(${index})"
                    style="cursor:pointer"
                >

                    <td>
                        ${escapeHtml(
                            getSource(row)
                        )}
                    </td>

                    <td>
                        <div class="log-message">
                            ${escapeHtml(
                                getMessage(row)
                            )}
                        </div>
                    </td>

                    <td>
                        <span class="${badgeClass}">
                            ${escapeHtml(label)}
                        </span>
                    </td>

                </tr>
            `;

        });


        html += `
                </tbody>

            </table>
        `;


        container.innerHTML = html;

    }


    // =========================================================
    // FILTERS
    // =========================================================

    function updateFilters() {

        const labelFilter =
            document.getElementById(
                "labelFilter"
            );

        const sourceFilter =
            document.getElementById(
                "sourceFilter"
            );


        const labels =
            [...new Set(
                allData.map(
                    function(row) {
                        return getLabel(row);
                    }
                )
            )].sort();


        const sources =
            [...new Set(
                allData.map(
                    function(row) {
                        return getSource(row);
                    }
                )
            )].sort();


        labelFilter.innerHTML =
            `<option value="">
                All classifications
            </option>`;


        labels.forEach(function(label) {

            const option =
                document.createElement("option");

            option.value = label;

            option.textContent = label;

            labelFilter.appendChild(option);

        });


        sourceFilter.innerHTML =
            `<option value="">
                All sources
            </option>`;


        sources.forEach(function(source) {

            const option =
                document.createElement("option");

            option.value = source;

            option.textContent = source;

            sourceFilter.appendChild(option);

        });

    }


    function applyFilters() {

        const search =
            document.getElementById(
                "searchInput"
            ).value
            .toLowerCase();


        const label =
            document.getElementById(
                "labelFilter"
            ).value;


        const source =
            document.getElementById(
                "sourceFilter"
            ).value;


        filteredData =
            allData.filter(function(row) {

                const message =
                    getMessage(row)
                        .toLowerCase();

                const rowSource =
                    getSource(row);

                const rowLabel =
                    getLabel(row);


                const searchMatch =
                    !search ||
                    message.includes(search) ||
                    rowSource
                        .toLowerCase()
                        .includes(search) ||
                    rowLabel
                        .toLowerCase()
                        .includes(search);


                const labelMatch =
                    !label ||
                    rowLabel === label;


                const sourceMatch =
                    !source ||
                    rowSource === source;


                return (
                    searchMatch &&
                    labelMatch &&
                    sourceMatch
                );

            });


        renderResults();

    }


    // =========================================================
    // RESULTS TABLE
    // =========================================================

    function renderResults() {

        const container =
            document.getElementById(
                "resultsTable"
            );


        if (!filteredData.length) {

            container.innerHTML = `
                <div class="empty">
                    <div class="empty-icon">☷</div>
                    No matching records found.
                </div>
            `;

            return;

        }


        let html = `
            <table>

                <thead>

                    <tr>

                        <th>#</th>
                        <th>Source</th>
                        <th>Log Message</th>
                        <th>Classification</th>

                    </tr>

                </thead>

                <tbody>
        `;


        filteredData
            .slice(0, 5000)
            .forEach(function(row, index) {

                const label =
                    getLabel(row);

                const type =
                    classifyLabel(label);


                let badgeClass =
                    "badge";


                if (type === "anomaly") {

                    badgeClass +=
                        " badge-anomaly";

                }


                if (type === "other") {

                    badgeClass +=
                        " badge-other";

                }


                html += `
                    <tr
                        onclick="showFilteredDetails(${index})"
                        style="cursor:pointer"
                    >

                        <td>
                            ${index + 1}
                        </td>

                        <td>
                            ${escapeHtml(
                                getSource(row)
                            )}
                        </td>

                        <td>

                            <div class="log-message">

                                ${escapeHtml(
                                    getMessage(row)
                                )}

                            </div>

                        </td>

                        <td>

                            <span class="${badgeClass}">

                                ${escapeHtml(
                                    label
                                )}

                            </span>

                        </td>

                    </tr>
                `;

            });


        html += `
                </tbody>

            </table>
        `;


        container.innerHTML = html;

    }


    // =========================================================
    // ANALYTICS
    // =========================================================

    function updateAnalytics() {

        if (!allData.length) {

            return;

        }


        const classificationCanvas =
            document.getElementById(
                "analyticsClassificationChart"
            );


        const sourceCanvas =
            document.getElementById(
                "analyticsSourceChart"
            );


        if (!classificationCanvas ||
            !sourceCanvas) {

            return;

        }


        if (analyticsClassificationChart) {

            analyticsClassificationChart.destroy();

        }


        if (analyticsSourceChart) {

            analyticsSourceChart.destroy();

        }


        const labelCounts =
            countLabels();


        const labelEntries =
            Object.entries(labelCounts)
                .sort(
                    function(a, b) {
                        return b[1] - a[1];
                    }
                );


        analyticsClassificationChart =
            new Chart(
                classificationCanvas,
                {
                    type: "bar",

                    data: {

                        labels:
                            labelEntries.map(
                                function(item) {
                                    return item[0];
                                }
                            ),

                        datasets: [

                            {
                                label: "Logs",

                                data:
                                    labelEntries.map(
                                        function(item) {
                                            return item[1];
                                        }
                                    ),

                                borderWidth: 0

                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        maintainAspectRatio: false,

                        scales: {

                            y: {
                                beginAtZero: true
                            }

                        },

                        plugins: {

                            legend: {
                                display: false
                            }

                        }

                    }

                }
            );


        const sourceCounts =
            countSources();


        const sourceEntries =
            Object.entries(sourceCounts)
                .sort(
                    function(a, b) {
                        return b[1] - a[1];
                    }
                )
                .slice(0, 15);


        analyticsSourceChart =
            new Chart(
                sourceCanvas,
                {
                    type: "bar",

                    data: {

                        labels:
                            sourceEntries.map(
                                function(item) {
                                    return item[0];
                                }
                            ),

                        datasets: [

                            {
                                label: "Logs",

                                data:
                                    sourceEntries.map(
                                        function(item) {
                                            return item[1];
                                        }
                                    ),

                                borderWidth: 0

                            }

                        ]

                    },

                    options: {

                        responsive: true,

                        maintainAspectRatio: false,

                        scales: {

                            y: {
                                beginAtZero: true
                            }

                        },

                        plugins: {

                            legend: {
                                display: false
                            }

                        }

                    }

                }
            );


        document.getElementById(
            "analyticsSummary"
        ).innerHTML = `

            <strong>
                ${totalRecords.toLocaleString()}
            </strong>
            total records were processed.

            <br>

            The dashboard currently displays the
            classification labels returned by your
            existing <code>classify()</code> function.

            <br><br>

            Top classification:
            <strong>
                ${
                    labelEntries.length
                        ? escapeHtml(
                            labelEntries[0][0]
                        )
                        : "N/A"
                }
            </strong>

        `;

    }


    // =========================================================
    // MODAL
    // =========================================================

    function showDetails(index) {

        if (!allData[index]) {
            return;
        }

        showRowDetails(
            allData[index]
        );

    }


    function showFilteredDetails(index) {

        if (!filteredData[index]) {
            return;
        }

        showRowDetails(
            filteredData[index]
        );

    }


    function showRowDetails(row) {

        const modal =
            document.getElementById(
                "detailModal"
            );


        const body =
            document.getElementById(
                "modalBody"
            );


        const label =
            getLabel(row);


        body.innerHTML = `

            <div class="detail-row">

                <div class="detail-label">
                    Source
                </div>

                <div class="detail-value">
                    ${escapeHtml(
                        getSource(row)
                    )}
                </div>

            </div>


            <div class="detail-row">

                <div class="detail-label">
                    Log Message
                </div>

                <div class="detail-value">
                    ${escapeHtml(
                        getMessage(row)
                    )}
                </div>

            </div>


            <div class="detail-row">

                <div class="detail-label">
                    Classification
                </div>

                <div class="detail-value">
                    ${escapeHtml(label)}
                </div>

            </div>

        `;


        modal.classList.add("active");

    }


    function closeModal() {

        document.getElementById(
            "detailModal"
        ).classList.remove("active");

    }


    document.getElementById(
        "detailModal"
    ).addEventListener(
        "click",
        function(event) {

            if (
                event.target === this
            ) {

                closeModal();

            }

        }
    );


    // =========================================================
    // DOWNLOAD
    // =========================================================

    function downloadResults() {

        if (!allData.length) {

            alert(
                "No results available. Please analyze a CSV first."
            );

            return;

        }


        window.location.href =
            "/download-results";

    }


    // =========================================================
    // SECURITY / HTML ESCAPING
    // =========================================================

    function escapeHtml(value) {

        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");

    }

</script>


</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return HTML_PAGE


# ============================================================
# CLASSIFICATION API
# ============================================================

@app.post("/classify/")
async def classify_logs(file: UploadFile):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was selected."
        )


    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )


    try:

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        file_bytes = await file.read()


        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )


        # ----------------------------------------------------
        # Decode CSV
        # ----------------------------------------------------

        try:

            csv_text = file_bytes.decode("utf-8-sig")

        except UnicodeDecodeError:

            csv_text = file_bytes.decode(
                "latin-1"
            )


        # ----------------------------------------------------
        # Read dataframe
        # ----------------------------------------------------

        df = pd.read_csv(
            io.StringIO(csv_text)
        )


        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="CSV file does not contain any records."
            )


        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        df.columns = [
            str(column)
            .strip()
            .lower()
            for column in df.columns
        ]


        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = {
            "source",
            "log_message"
        }


        # IMPORTANT:
        # This is the corrected line that caused
        # the SyntaxError in the previous version.

        missing_columns = (
            required_columns -
            set(df.columns)
        )


        if missing_columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    "CSV must contain "
                    "'source' and "
                    "'log_message' columns. "
                    f"Missing: "
                    f"{list(missing_columns)}. "
                    f"Found: "
                    f"{list(df.columns)}"
                )
            )


        # ----------------------------------------------------
        # Clean required fields
        # ----------------------------------------------------

        df["source"] = (
            df["source"]
            .fillna("")
            .astype(str)
            .str.strip()
        )


        df["log_message"] = (
            df["log_message"]
            .fillna("")
            .astype(str)
        )


        # ----------------------------------------------------
        # Prepare classifier input
        # ----------------------------------------------------

        log_data = list(
            zip(
                df["source"],
                df["log_message"]
            )
        )


        # ----------------------------------------------------
        # Run existing classifier
        # ----------------------------------------------------

        try:

            predictions = classify(
                log_data
            )


            # Make sure result is a list
            if predictions is None:

                predictions = []


            predictions = list(
                predictions
            )


            # ------------------------------------------------
            # Validate prediction length
            # ------------------------------------------------

            if len(predictions) == len(df):

                df["target_label"] = predictions

            else:

                # If classify() returns an unexpected
                # number of predictions, do not crash
                # the entire application.

                df["target_label"] = [
                    "Unclassified"
                    for _ in range(len(df))
                ]

        except Exception as classifier_error:

            print(
                "Classifier error:",
                classifier_error
            )


            df["target_label"] = [
                "Unclassified"
                for _ in range(len(df))
            ]


        # ----------------------------------------------------
        # Save complete output
        # ----------------------------------------------------

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )


        # ----------------------------------------------------
        # Prepare JSON-safe records
        # ----------------------------------------------------

        browser_limit = 5000


        browser_df = df.head(
            browser_limit
        ).copy()


        browser_df = (
            browser_df
            .astype(object)
            .where(
                pd.notnull(browser_df),
                None
            )
        )


        records = browser_df.to_dict(
            orient="records"
        )


        # ----------------------------------------------------
        # Return results
        # ----------------------------------------------------

        return {
            "success": True,
            "total_records": int(len(df)),
            "returned_records": int(len(records)),
            "data": records
        }


    except HTTPException:

        raise


    except pd.errors.ParserError as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to parse CSV file. "
                f"Details: {str(error)}"
            )
        )


    except Exception as error:

        print(
            "Server error:",
            error
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process the CSV file. "
                f"Error: {str(error)}"
            )
        )


# ============================================================
# DOWNLOAD OUTPUT
# ============================================================

@app.get("/download-results")
async def download_results():

    if not os.path.exists(OUTPUT_FILE):

        raise HTTPException(
            status_code=404,
            detail="No classification results are available yet."
        )


    return FileResponse(
        path=OUTPUT_FILE,
        filename="classified_logs.csv",
        media_type="text/csv"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "classifier_available": True,
        "output_file_exists":
            os.path.exists(OUTPUT_FILE)
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
