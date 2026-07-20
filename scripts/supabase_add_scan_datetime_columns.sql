-- Add readable scan date/time columns to an existing Supabase project.
-- Run this once in Supabase SQL Editor after the original schema is already installed.

alter table public.scans
    add column if not exists scan_date text not null default '',
    add column if not exists scan_time text not null default '',
    add column if not exists scan_datetime_local text not null default '',
    add column if not exists scan_timezone text not null default '';

update public.scans
set
    scan_date = coalesce(nullif(scan_date, ''), left(ts, 10)),
    scan_time = coalesce(nullif(scan_time, ''), substr(replace(ts, 'T', ' '), 12, 8)),
    scan_datetime_local = coalesce(
        nullif(scan_datetime_local, ''),
        trim(left(replace(ts, 'T', ' '), 19) || ' UTC')
    ),
    scan_timezone = coalesce(nullif(scan_timezone, ''), 'UTC')
where scan_date = ''
   or scan_time = ''
   or scan_datetime_local = ''
   or scan_timezone = '';

create index if not exists scans_scan_date_idx on public.scans (scan_date);
create index if not exists scans_scan_datetime_local_idx on public.scans (scan_datetime_local desc);

create or replace function public.reset_safesandesh_identity_sequences()
returns void
language plpgsql
security definer
set search_path = public
as $$
declare
    scan_max bigint;
begin
    select coalesce(max(id), 0) into scan_max from public.scans;
    perform setval(pg_get_serial_sequence('public.scans', 'id'), greatest(scan_max, 1), scan_max > 0);
end;
$$;

grant execute on function public.reset_safesandesh_identity_sequences() to service_role;
