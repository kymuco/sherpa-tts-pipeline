param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetSourcesCsv,

    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [Parameter(Mandatory = $true)]
    [string]$ProcessedRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [ValidateSet("Copy", "HardLink")]
    [string]$Mode = "Copy"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-NormalizedKey {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    return $Value.Normalize([Text.NormalizationForm]::FormKC).ToLowerInvariant()
}

function Get-FileMap {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    if (-not (Test-Path -LiteralPath $Root)) {
        throw "Directory not found: $Root"
    }

    $map = @{}
    foreach ($file in Get-ChildItem -LiteralPath $Root -File) {
        $key = Get-NormalizedKey -Value $file.Name
        if ($map.ContainsKey($key)) {
            throw "Duplicate normalized filename in ${Root}: $($file.Name)"
        }
        $map[$key] = $file
    }

    return $map
}

function Save-File {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$Source,

        [Parameter(Mandatory = $true)]
        [string]$Destination,

        [Parameter(Mandatory = $true)]
        [string]$WriteMode
    )

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Force
    }

    if ($WriteMode -eq "HardLink") {
        New-Item -ItemType HardLink -Path $Destination -Target $Source.FullName | Out-Null
        return
    }

    Copy-Item -LiteralPath $Source.FullName -Destination $Destination -Force
}

if (-not (Test-Path -LiteralPath $DatasetSourcesCsv)) {
    throw "Dataset sources file not found: $DatasetSourcesCsv"
}

$datasetRows = Import-Csv -Path $DatasetSourcesCsv -Encoding UTF8
$sourceMap = Get-FileMap -Root $SourceRoot
$processedMap = Get-FileMap -Root $ProcessedRoot

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$manifestRows = New-Object System.Collections.Generic.List[object]
$pendingCount = 0
$processedCount = 0
$missingCount = 0

foreach ($row in $datasetRows) {
    $leafName = Split-Path ([string]$row.source_file) -Leaf
    $key = Get-NormalizedKey -Value $leafName
    $status = ""
    $sourcePath = ""
    $processedPath = ""
    $pendingBatchPath = ""

    if ($processedMap.ContainsKey($key)) {
        $status = "processed"
        $processedPath = $processedMap[$key].FullName
        $processedCount += 1
    }
    elseif ($sourceMap.ContainsKey($key)) {
        $status = "pending"
        $sourceFile = $sourceMap[$key]
        $sourcePath = $sourceFile.FullName
        $pendingBatchPath = Join-Path $OutDir $sourceFile.Name
        Save-File -Source $sourceFile -Destination $pendingBatchPath -WriteMode $Mode
        $pendingCount += 1
    }
    else {
        $status = "missing"
        $missingCount += 1
    }

    $manifestRows.Add([PSCustomObject][ordered]@{
        dataset_source_name = $leafName
        status = $status
        source_path = $sourcePath
        processed_path = $processedPath
        pending_batch_path = $pendingBatchPath
        kept_chunks = $row.kept_chunks
    }) | Out-Null
}

$manifestPath = Join-Path $OutDir "uvr_manifest.csv"
$manifestRows | Export-Csv -Path $manifestPath -NoTypeInformation -Encoding UTF8

[PSCustomObject]@{
    dataset_sources = $datasetRows.Count
    processed = $processedCount
    pending = $pendingCount
    missing = $missingCount
    out_dir = $OutDir
    manifest = $manifestPath
    mode = $Mode
} | ConvertTo-Json -Depth 3
