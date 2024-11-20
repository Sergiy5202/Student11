--
-- PostgreSQL database dump
--

-- Dumped from database version 16.1
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

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: sensors; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sensors (
    sensor_id integer NOT NULL,
    location character varying(255) NOT NULL,
    type character varying(100) NOT NULL,
    status boolean DEFAULT true NOT NULL
);


ALTER TABLE public.sensors OWNER TO postgres;

--
-- Name: sensors_sensor_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.sensors_sensor_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.sensors_sensor_id_seq OWNER TO postgres;

--
-- Name: sensors_sensor_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.sensors_sensor_id_seq OWNED BY public.sensors.sensor_id;


--
-- Name: sensors sensor_id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sensors ALTER COLUMN sensor_id SET DEFAULT nextval('public.sensors_sensor_id_seq'::regclass);


--
-- Data for Name: sensors; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sensors (sensor_id, location, type, status) FROM stdin;
1	Building A - Room 101	Temperature	t
2	Building B - Room 202	Humidity	t
3	Building C - Hall	Motion	f
\.


--
-- Name: sensors_sensor_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.sensors_sensor_id_seq', 3, true);


--
-- Name: sensors sensors_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sensors
    ADD CONSTRAINT sensors_pkey PRIMARY KEY (sensor_id);


--
-- PostgreSQL database dump complete
--

