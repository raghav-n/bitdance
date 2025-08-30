import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import { LogSegment } from "@/types/log-cr-types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const toggleSetItem = <T>(set: Set<T>, item: T, setFunction: React.Dispatch<React.SetStateAction<Set<T>>>): void => {
  const newSet = new Set(set);
  if (newSet.has(item)) {
    newSet.delete(item);
  } else {
    newSet.add(item);
  }
  setFunction(newSet);
}

// Date formatting and validation utilities
export const formatDateInput = (value: string): string => {
  // Remove any non-digit characters
  const digits = value.replace(/\D/g, '');
  
  // Format as dd/mm/yyyy
  if (digits.length <= 2) {
    return digits;
  } else if (digits.length <= 4) {
    return `${digits.substring(0, 2)}/${digits.substring(2)}`;
  } else {
    return `${digits.substring(0, 2)}/${digits.substring(2, 4)}/${digits.substring(4, 8)}`;
  }
};

export const validateDateFormat = (value: string): boolean => {
  if (value === '') return true;
  
  // Check if the value matches dd/mm/yyyy format
  const dateRegex = /^(\d{2})\/(\d{2})\/(\d{4})$/;
  if (!dateRegex.test(value)) {
    return false;
  }
  
  // Validate date parts
  const match = dateRegex.exec(value);
  if (match) {
    const day = parseInt(match[1]);
    const month = parseInt(match[2]);
    const year = parseInt(match[3]);
    
    return !(day < 1 || day > 31 || month < 1 || month > 12 || year < 1900 || year > 2100);
  }
  
  return false;
};

// Filter log segments by criteria
export function filterLogSegments(
  logSegments: LogSegment[], 
  filters: {
    ablr?: string
    user?: string
    hostname?: string
    timeFrom?: string
    timeTo?: string
  }
): LogSegment[] {
  return logSegments.filter(segment => {
    // Filter by ABLR
    if (filters.ablr && filters.ablr !== 'all' && !segment.ablr_use_cases.includes(filters.ablr)) {
      return false;
    }
    
    // Filter by user
    if (filters.user && filters.user !== 'all' && !segment.users.some(user => 
      user.toLowerCase().includes(filters.user!.toLowerCase())
    )) {
      return false;
    }
    
    // Filter by hostname
    if (filters.hostname && filters.hostname !== 'all' && !segment.hostnames.some(hostname => 
      hostname.toLowerCase().includes(filters.hostname!.toLowerCase())
    )) {
      return false;
    }
    
    // Filter by time range
    if (filters.timeFrom || filters.timeTo) {
      const matchTime = new Date(segment.log_start_time).getTime();
      
      if (filters.timeFrom) {
        // Parse dd/mm/yyyy format
        const fromParts = filters.timeFrom.split('/');
        if (fromParts.length === 3) {
          const day = parseInt(fromParts[0]);
          const month = parseInt(fromParts[1]) - 1; // JS months are 0-indexed
          const year = parseInt(fromParts[2]);
          
          if (!isNaN(day) && !isNaN(month) && !isNaN(year)) {
            const fromTime = new Date(year, month, day).getTime();
            if (matchTime < fromTime) {
              return false;
            }
          }
        }
      }
      
      if (filters.timeTo) {
        // Parse dd/mm/yyyy format
        const toParts = filters.timeTo.split('/');
        if (toParts.length === 3) {
          const day = parseInt(toParts[0]);
          const month = parseInt(toParts[1]) - 1; // JS months are 0-indexed
          const year = parseInt(toParts[2]);
          
          if (!isNaN(day) && !isNaN(month) && !isNaN(year)) {
            // Set time to end of day
            const toTime = new Date(year, month, day, 23, 59, 59).getTime();
            if (matchTime > toTime) {
              return false;
            }
          }
        }
      }
    }
    
    return true;
  });
}

export function createAllMatchesArray(match: LogSegment, isMatched: boolean): {
  match: string;
  description: string;
  match_score?: number;
}[] {
  let allMatches: {
    match: string;
    description: string;
    match_score?: number;
  }[] = [];
  
  if (isMatched) {
    allMatches.push({
      match: match.match || "",
      description: match.match_description,
      match_score: match.match_score
    });
  }

  const other_matches = []
  for (let i = 1; i <= 5; i++) {
    if (match.other_matches.hasOwnProperty(`other_match_${i}`) && match.other_matches.hasOwnProperty(`other_match_description_${i}`)) {
      const match_id = match.other_matches[`other_match_${i}`]
      const match_description = match.other_matches[`other_match_description_${i}`]
      if (match_id && match_description) {
      other_matches.push({
        match: match_id,
        description: match_description
        })
      }
    }
  }
  
  allMatches = allMatches.concat([
    ...other_matches,
    {
      match: "N/A",
      description: "This log entry does not match any known command or pattern",
    }
  ]);
  
  // remove empty matches and duplicates
  return allMatches
    .filter((match, index, self) => 
      index === self.findIndex(m => m.match === match.match)
    );
}