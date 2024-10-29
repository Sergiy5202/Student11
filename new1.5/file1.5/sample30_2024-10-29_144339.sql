--
-- PostgreSQL database dump
--

-- Dumped from database version 10.23
-- Dumped by pg_dump version 16.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: postgres
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO postgres;

SET default_tablespace = '';

--
-- Name: sensors_data; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sensors_data (
    data_id integer NOT NULL,
    sensor_id integer NOT NULL,
    temperature numeric(5,2),
    humidity numeric(5,2),
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.sensors_data OWNER TO postgres;

--
-- Name: COLUMN sensors_data.sensor_id; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.sensors_data.sensor_id IS 'comment';


--
-- Name: sensors_data_data_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.sensors_data_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sensors_data_data_id_seq OWNER TO postgres;

--
-- Name: sensors_data_data_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.sensors_data_data_id_seq OWNED BY public.sensors_data.data_id;


--
-- Name: sensors_data data_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sensors_data ALTER COLUMN data_id SET DEFAULT nextval('public.sensors_data_data_id_seq'::regclass);


--
-- Data for Name: sensors_data; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sensors_data (data_id, sensor_id, temperature, humidity, "timestamp") FROM stdin;
1	1	23.45	45.20	2023-10-01 14:25:00
2	1	22.98	47.50	2023-10-01 15:00:00
3	2	19.34	40.00	2023-10-01 15:15:00
4	3	21.56	42.10	2023-10-01 16:30:00
5	2	20.76	43.50	2023-10-01 17:00:00
6	1	24.05	46.70	2023-10-01 17:45:00
\.


--
-- Name: sensors_data_data_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.sensors_data_data_id_seq', 6, true);


--
-- Name: sensors_data sensors_data_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sensors_data
    ADD CONSTRAINT sensors_data_pkey PRIMARY KEY (data_id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

